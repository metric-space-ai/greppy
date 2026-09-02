//! Local JSON-RPC control transport for hosted agent sessions.

#![cfg(unix)]

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};

use crate::agent_tui::SessionStore;

const MAX_LINE: usize = 1024 * 1024;
const ACCEPT_POLL: Duration = Duration::from_millis(10);

pub type ConnId = u64;

#[derive(Debug)]
pub enum Incoming {
    Connected {
        conn: ConnId,
    },
    Request {
        conn: ConnId,
        id: Value,
        method: String,
        params: Value,
    },
    Disconnected {
        conn: ConnId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for RpcError {}

enum WireIncoming {
    Connected(ConnId),
    Line(ConnId, Vec<u8>),
    Oversized(ConnId),
    Disconnected(ConnId),
}

struct Connection {
    writer: UnixStream,
    subscribed: bool,
    pending: Vec<u8>,
}

pub struct ControlServer {
    path: PathBuf,
    incoming: Receiver<WireIncoming>,
    connections: Arc<Mutex<HashMap<ConnId, Connection>>>,
    stop: Arc<AtomicBool>,
    accept: Option<JoinHandle<()>>,
}

impl ControlServer {
    pub fn bind(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let listener = UnixListener::bind(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;

        let (tx, incoming) = mpsc::channel();
        let connections = Arc::new(Mutex::new(HashMap::new()));
        let accept_connections = Arc::clone(&connections);
        let stop = Arc::new(AtomicBool::new(false));
        let accept_stop = Arc::clone(&stop);
        let next_id = Arc::new(AtomicU64::new(1));
        let accept = thread::Builder::new()
            .name("greppy-agent-control-accept".to_string())
            .spawn(move || {
                while !accept_stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let conn = next_id.fetch_add(1, Ordering::Relaxed);
                            let reader = match stream.try_clone() {
                                Ok(reader) => reader,
                                Err(_) => continue,
                            };
                            if stream.set_nonblocking(true).is_err() {
                                continue;
                            }
                            if let Ok(mut all) = accept_connections.lock() {
                                all.insert(
                                    conn,
                                    Connection {
                                        writer: stream,
                                        subscribed: false,
                                        pending: Vec::new(),
                                    },
                                );
                            }
                            let _ = tx.send(WireIncoming::Connected(conn));
                            let reader_tx = tx.clone();
                            let _ = thread::Builder::new()
                                .name(format!("greppy-agent-control-{conn}"))
                                .spawn(move || read_connection(conn, reader, reader_tx));
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(ACCEPT_POLL);
                        }
                        Err(_) => break,
                    }
                }
            })?;

        Ok(Self {
            path: path.to_path_buf(),
            incoming,
            connections,
            stop,
            accept: Some(accept),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn poll(&mut self) -> Vec<Incoming> {
        self.flush_connections();
        let mut out = Vec::new();
        loop {
            match self.incoming.try_recv() {
                Ok(WireIncoming::Connected(conn)) => out.push(Incoming::Connected { conn }),
                Ok(WireIncoming::Disconnected(conn) | WireIncoming::Oversized(conn)) => {
                    self.disconnect(conn);
                    out.push(Incoming::Disconnected { conn });
                }
                Ok(WireIncoming::Line(conn, line)) => match parse_request(&line) {
                    Ok((id, method, params)) => {
                        if method == "session/subscribe" {
                            if let Ok(mut connections) = self.connections.lock() {
                                if let Some(connection) = connections.get_mut(&conn) {
                                    connection.subscribed = true;
                                }
                            }
                        }
                        out.push(Incoming::Request {
                            conn,
                            id,
                            method,
                            params,
                        });
                    }
                    Err(error) => self.reply(conn, Value::Null, Err(error)),
                },
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        out
    }

    pub fn reply(&mut self, conn: ConnId, id: Value, result: Result<Value, RpcError>) {
        let value = match result {
            Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
            Err(error) => json!({
                "jsonrpc":"2.0",
                "id":id,
                "error":{"code":error.code,"message":error.message},
            }),
        };
        self.send_value(conn, &value);
    }

    pub fn broadcast(&mut self, event: &Value) {
        let value = json!({"jsonrpc":"2.0","method":"event","params":event});
        let bytes = encoded_line(&value);
        let ids = self
            .connections
            .lock()
            .map(|connections| {
                connections
                    .iter()
                    .filter_map(|(id, connection)| connection.subscribed.then_some(*id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for id in ids {
            self.send_bytes(id, &bytes);
        }
    }

    fn send_value(&mut self, conn: ConnId, value: &Value) {
        self.send_bytes(conn, &encoded_line(value));
    }

    fn send_bytes(&mut self, conn: ConnId, bytes: &[u8]) {
        let mut disconnect = false;
        if let Ok(mut connections) = self.connections.lock() {
            if let Some(connection) = connections.get_mut(&conn) {
                if !connection.pending.is_empty() {
                    connection.pending.extend_from_slice(bytes);
                } else {
                    match connection.writer.write(bytes) {
                        Ok(written) if written < bytes.len() => {
                            connection.pending.extend_from_slice(&bytes[written..]);
                        }
                        Ok(_) => {}
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            connection.pending.extend_from_slice(bytes);
                        }
                        Err(_) => disconnect = true,
                    }
                }
                disconnect |= connection.pending.len() > MAX_LINE;
            }
        }
        if disconnect {
            self.disconnect(conn);
        }
    }

    fn flush_connections(&mut self) {
        let mut dead = Vec::new();
        if let Ok(mut connections) = self.connections.lock() {
            for (id, connection) in connections.iter_mut() {
                while !connection.pending.is_empty() {
                    match connection.writer.write(&connection.pending) {
                        Ok(0) => {
                            dead.push(*id);
                            break;
                        }
                        Ok(written) => {
                            connection.pending.drain(..written);
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                        Err(_) => {
                            dead.push(*id);
                            break;
                        }
                    }
                }
            }
        }
        for id in dead {
            self.disconnect(id);
        }
    }

    fn disconnect(&mut self, conn: ConnId) {
        if let Ok(mut connections) = self.connections.lock() {
            if let Some(connection) = connections.remove(&conn) {
                let _ = connection.writer.shutdown(std::net::Shutdown::Both);
            }
        }
    }
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
        if let Ok(mut connections) = self.connections.lock() {
            for (_, connection) in connections.drain() {
                let _ = connection.writer.shutdown(std::net::Shutdown::Both);
            }
        }
        let _ = fs::remove_file(&self.path);
    }
}

fn parse_request(line: &[u8]) -> Result<(Value, String, Value), RpcError> {
    let value: Value =
        serde_json::from_slice(line).map_err(|_| RpcError::new(-32700, "parse error"))?;
    let object = value
        .as_object()
        .ok_or_else(|| RpcError::new(-32600, "invalid request"))?;
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(RpcError::new(-32600, "invalid request"));
    }
    let id = object
        .get("id")
        .filter(|id| id.is_string() || id.as_i64().is_some() || id.as_u64().is_some())
        .cloned()
        .ok_or_else(|| RpcError::new(-32600, "invalid request"))?;
    let method = object
        .get("method")
        .and_then(Value::as_str)
        .filter(|method| !method.is_empty())
        .ok_or_else(|| RpcError::new(-32600, "invalid request"))?
        .to_string();
    let params = object.get("params").cloned().unwrap_or_else(|| json!({}));
    if !params.is_object() {
        return Err(RpcError::new(-32602, "invalid params"));
    }
    Ok((id, method, params))
}

fn encoded_line(value: &Value) -> Vec<u8> {
    let mut bytes = value.to_string().into_bytes();
    bytes.push(b'\n');
    bytes
}

fn read_connection(conn: ConnId, mut stream: UnixStream, tx: mpsc::Sender<WireIncoming>) {
    let mut pending = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => {
                pending.extend_from_slice(&chunk[..count]);
                while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
                    if newline > MAX_LINE {
                        let _ = tx.send(WireIncoming::Oversized(conn));
                        return;
                    }
                    let mut line: Vec<u8> = pending.drain(..=newline).collect();
                    line.pop();
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    let _ = tx.send(WireIncoming::Line(conn, line));
                }
                if pending.len() > MAX_LINE {
                    let _ = tx.send(WireIncoming::Oversized(conn));
                    return;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL);
            }
            Err(_) => break,
        }
    }
    let _ = tx.send(WireIncoming::Disconnected(conn));
}

pub struct ControlClient {
    writer: UnixStream,
    reader: BufReader<UnixStream>,
    next_id: u64,
    events: VecDeque<Value>,
}

impl ControlClient {
    pub fn connect(path: &Path) -> io::Result<Self> {
        let writer = UnixStream::connect(path)?;
        let reader = BufReader::new(writer.try_clone()?);
        Ok(Self {
            writer,
            reader,
            next_id: 1,
            events: VecDeque::new(),
        })
    }

    pub fn call(&mut self, method: &str, params: Value) -> Result<Value, RpcError> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let request = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        self.writer
            .write_all(&encoded_line(&request))
            .and_then(|_| self.writer.flush())
            .map_err(io_rpc_error)?;
        loop {
            let value = self.read_value().map_err(io_rpc_error)?;
            if value.get("method").and_then(Value::as_str) == Some("event") {
                if let Some(event) = value.get("params") {
                    self.events.push_back(event.clone());
                }
                continue;
            }
            if value.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = value.get("error") {
                return Err(RpcError::new(
                    error.get("code").and_then(Value::as_i64).unwrap_or(-32000),
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("control request failed"),
                ));
            }
            return value
                .get("result")
                .cloned()
                .ok_or_else(|| RpcError::new(-32603, "invalid response"));
        }
    }

    pub fn subscribe(&mut self) -> Result<(), RpcError> {
        let result = self.call("session/subscribe", json!({}))?;
        if result.get("subscribed").and_then(Value::as_bool) == Some(true) {
            Ok(())
        } else {
            Err(RpcError::new(-32603, "invalid subscribe response"))
        }
    }

    pub fn next_event(&mut self, timeout: Duration) -> io::Result<Option<Value>> {
        if let Some(event) = self.events.pop_front() {
            return Ok(Some(event));
        }
        self.reader.get_ref().set_read_timeout(Some(timeout))?;
        loop {
            match self.read_value() {
                Ok(value) => {
                    if value.get("method").and_then(Value::as_str) == Some("event") {
                        return Ok(value.get("params").cloned());
                    }
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn read_value(&mut self) -> io::Result<Value> {
        let mut line = Vec::new();
        let count = self.reader.read_until(b'\n', &mut line)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "control socket closed",
            ));
        }
        if line.len() > MAX_LINE + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "control line exceeds 1 MiB",
            ));
        }
        serde_json::from_slice(&line)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

fn io_rpc_error(error: io::Error) -> RpcError {
    RpcError::new(-32001, error.to_string())
}

pub fn socket_path_for(store: &SessionStore, session_id: &str) -> PathBuf {
    store.path_for(session_id).with_extension("sock")
}

pub fn is_live(path: &Path) -> bool {
    UnixStream::connect(path).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_socket(tag: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "greppy-control-{tag}-{}-{nonce}.sock",
            std::process::id()
        ))
    }

    fn wait_for_request(server: &mut ControlServer) -> (ConnId, Value, String, Value) {
        for _ in 0..200 {
            for incoming in server.poll() {
                if let Incoming::Request {
                    conn,
                    id,
                    method,
                    params,
                } = incoming
                {
                    return (conn, id, method, params);
                }
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("request not received")
    }

    #[test]
    fn socket_is_private_and_removed_on_drop() {
        let path = temp_socket("mode");
        let server = ControlServer::bind(&path).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        drop(server);
        assert!(!path.exists());
    }

    #[test]
    fn call_subscribe_and_broadcast() {
        let path = temp_socket("roundtrip");
        let mut server = ControlServer::bind(&path).unwrap();
        let path_for_client = path.clone();
        let client = thread::spawn(move || {
            let mut client = ControlClient::connect(&path_for_client).unwrap();
            assert_eq!(
                client.call("session/describe", json!({})).unwrap(),
                json!({"ok":true})
            );
            client.subscribe().unwrap();
            client.next_event(Duration::from_secs(2)).unwrap().unwrap()
        });
        let (conn, id, method, _) = wait_for_request(&mut server);
        assert_eq!(method, "session/describe");
        server.reply(conn, id, Ok(json!({"ok":true})));
        let (conn, id, method, _) = wait_for_request(&mut server);
        assert_eq!(method, "session/subscribe");
        server.reply(conn, id, Ok(json!({"subscribed":true})));
        server.broadcast(&json!({"type":"phase","phase":"idle"}));
        assert_eq!(client.join().unwrap()["phase"], "idle");
    }

    #[test]
    fn malformed_request_gets_parse_error_and_connection_survives() {
        let path = temp_socket("parse");
        let mut server = ControlServer::bind(&path).unwrap();
        let mut stream = UnixStream::connect(&path).unwrap();
        stream.write_all(b"not-json\n").unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        for _ in 0..100 {
            server.poll();
            let _ = reader
                .get_ref()
                .set_read_timeout(Some(Duration::from_millis(10)));
            let mut line = String::new();
            if reader.read_line(&mut line).is_ok() && !line.is_empty() {
                let value: Value = serde_json::from_str(&line).unwrap();
                assert_eq!(value["error"]["code"], -32700);
                break;
            }
        }
        stream
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ok\",\"params\":{}}\n")
            .unwrap();
        let (_, _, method, _) = wait_for_request(&mut server);
        assert_eq!(method, "ok");
    }
}

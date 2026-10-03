use super::*;
use crate::protocol::ToolChoice;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

fn request() -> ModelRequest {
    ModelRequest {
        model: "fixture-model".into(),
        system: None,
        messages: vec![],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: 8,
    }
}

fn read_request(stream: &mut TcpStream) {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        if line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
}

fn cancelled_at_blocked_read(send_headers: bool) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (closed_tx, closed_rx) = mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_request(&mut stream);
        if send_headers {
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n: fixture ping\n\n").unwrap();
            stream.flush().unwrap();
        }
        ready_tx.send(()).unwrap();
        // The response intentionally stalls. Cancellation must close this socket.
        let closed = matches!(stream.read(&mut [0u8; 1]), Ok(0));
        closed_tx.send(closed).unwrap();
    });
    let cancel = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancel);
    let cancel_job = thread::spawn(move || {
        ready_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        thread::sleep(Duration::from_millis(50));
        signal.store(true, Ordering::Relaxed);
    });
    let client = Client::new(&format!("http://{address}"), "fixture-model");
    let start = Instant::now();
    let result = client.stream_turn_interruptible(&request(), &mut |_| {}, &cancel);
    assert_eq!(result, Err(ClientError::Cancelled));
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "cancellation remained blocked"
    );
    cancel_job.join().unwrap();
    assert!(
        closed_rx.recv_timeout(Duration::from_secs(3)).unwrap(),
        "model socket stayed open"
    );
    server.join().unwrap();
}

#[test]
fn cancel_while_waiting_for_response_headers_closes_request() {
    cancelled_at_blocked_read(false);
}

#[test]
fn cancel_while_waiting_for_sse_body_closes_request() {
    cancelled_at_blocked_read(true);
}

#[test]
fn completed_interruptible_turn_preserves_text_and_usage() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_request(&mut stream);
        let body = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"fixture-model\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Grüße\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
        stream.write_all(body.as_bytes()).unwrap();
    });
    let client = Client::new(&format!("http://{address}"), "fixture-model");
    let mut events = Vec::new();
    let result = client
        .stream_turn_interruptible(
            &request(),
            &mut |event| events.push(event),
            &AtomicBool::new(false),
        )
        .unwrap();
    server.join().unwrap();
    assert_eq!(
        result.message.content,
        vec![ContentPart::Text {
            text: "Grüße".into()
        }]
    );
    assert_eq!(result.usage.input_tokens, 3);
    assert_eq!(result.usage.output_tokens, 2);
    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert!(events
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "Grüße")));
}

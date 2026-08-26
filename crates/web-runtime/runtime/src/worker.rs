use crate::protocol::{read_message, write_message, Message, WorkerKind};
use std::io::{self, Read, Write};

pub fn run_worker<R, W, T>(
    worker: WorkerKind,
    runtime: T,
    reader: &mut R,
    writer: &mut W,
) -> io::Result<()>
where
    R: Read,
    W: Write,
{
    match read_message(reader)? {
        Message::Hello {
            worker: requested, ..
        } if requested == worker => {}
        unexpected => return Err(unexpected_message("Hello", worker, unexpected)),
    }

    write_message(writer, &Message::ready(worker))?;

    match read_message(reader)? {
        Message::Shutdown { .. } => {}
        unexpected => return Err(unexpected_message("Shutdown", worker, unexpected)),
    }

    drop(runtime);
    write_message(writer, &Message::shutdown_ack(worker))
}

fn unexpected_message(expected: &str, worker: WorkerKind, actual: Message) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{worker:?} worker expected {expected}, received {actual:?}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::Cursor;
    use std::rc::Rc;

    struct DropMarker(Rc<Cell<bool>>);

    impl Drop for DropMarker {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    #[test]
    fn acknowledges_shutdown_only_after_runtime_is_dropped() {
        let dropped = Rc::new(Cell::new(false));
        let mut input = Vec::new();
        write_message(&mut input, &Message::hello(WorkerKind::Controller)).unwrap();
        write_message(&mut input, &Message::shutdown()).unwrap();
        let mut output = Vec::new();

        run_worker(
            WorkerKind::Controller,
            DropMarker(Rc::clone(&dropped)),
            &mut Cursor::new(input),
            &mut output,
        )
        .unwrap();

        assert!(dropped.get());
        let mut output = Cursor::new(output);
        assert_eq!(
            read_message(&mut output).unwrap(),
            Message::ready(WorkerKind::Controller)
        );
        assert_eq!(
            read_message(&mut output).unwrap(),
            Message::shutdown_ack(WorkerKind::Controller)
        );
    }

    #[test]
    fn rejects_a_hello_for_the_other_worker() {
        let mut input = Vec::new();
        write_message(&mut input, &Message::hello(WorkerKind::Content)).unwrap();

        let error = run_worker(
            WorkerKind::Controller,
            (),
            &mut Cursor::new(input),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}

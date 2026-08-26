use deno_core::{JsRuntime, RuntimeOptions};
use servo::{Servo, ServoBuilder};

fn construct_deno() -> JsRuntime {
    let mut javascript = JsRuntime::new(RuntimeOptions::default());
    javascript
        .execute_script("<phase1-probe>", "globalThis.__phase1_probe = 1 + 1;")
        .expect("deno_core must execute JavaScript");
    javascript
}

fn construct_servo() -> Servo {
    ServoBuilder::default().build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).ok_or(
        "missing required mode: deno-only | servo-only | deno-then-servo | servo-then-deno",
    )?;

    match mode.as_str() {
        "deno-only" => {
            let _javascript = construct_deno();
            println!("phase1_probe.deno_core=ready");
            println!("phase1_probe.process=ready");
        }
        "servo-only" => {
            let _servo = construct_servo();
            println!("phase1_probe.servo=ready");
            println!("phase1_probe.process=ready");
        }
        "deno-then-servo" => {
            let _javascript = construct_deno();
            println!("phase1_probe.deno_core=ready");
            let _servo = construct_servo();
            println!("phase1_probe.servo=ready");
            println!("phase1_probe.process=ready");
        }
        "servo-then-deno" => {
            let _servo = construct_servo();
            println!("phase1_probe.servo=ready");
            let _javascript = construct_deno();
            println!("phase1_probe.deno_core=ready");
            println!("phase1_probe.process=ready");
        }
        other => {
            return Err(format!(
                "unknown mode {other:?}; expected deno-only | servo-only | deno-then-servo | servo-then-deno"
            )
            .into());
        }
    }

    Ok(())
}

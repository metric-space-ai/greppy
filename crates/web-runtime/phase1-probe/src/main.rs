use deno_core::{JsRuntime, RuntimeOptions};
use servo::{Servo, ServoBuilder};

const REGEX_INTL: &str = r#"
const regexOk = /foo+/i.test("FOOO") && "abc".replace(/b/, "B") === "aBc" && "abab".match(/a(b)/)[1] === "b";
if (!regexOk) throw new Error("regex stress failed");
const intlNumber = new Intl.NumberFormat("en-US").format(1234.5);
if (typeof intlNumber !== "string" || intlNumber.length === 0) throw new Error("Intl.NumberFormat failed");
const intlDate = new Intl.DateTimeFormat("en-US", { timeZone: "UTC" }).format(new Date(0));
if (typeof intlDate !== "string" || intlDate.length === 0) throw new Error("Intl.DateTimeFormat failed");
const compared = "é".localeCompare("e", "en");
if (typeof compared !== "number") throw new Error("localeCompare failed");
"#;

fn construct_deno() -> JsRuntime {
    construct_deno_with_stress(true)
}

fn construct_deno_plain() -> JsRuntime {
    construct_deno_with_stress(false)
}

fn construct_deno_with_stress(regex_intl: bool) -> JsRuntime {
    let mut javascript = JsRuntime::new(RuntimeOptions::default());
    javascript
        .execute_script("<phase1-probe>", "globalThis.__phase1_probe = 1 + 1;")
        .expect("deno_core must execute JavaScript");
    if regex_intl {
        javascript
            .execute_script("<phase1-probe-regex-intl>", REGEX_INTL)
            .expect("deno_core regex/Intl stress must execute");
    }
    javascript
}

fn construct_servo() -> Servo {
    ServoBuilder::default().build()
}

fn drop_deno(javascript: JsRuntime) {
    drop(javascript);
}

fn drop_servo(servo: Servo) {
    drop(servo);
}

fn run_deno_only() {
    let javascript = construct_deno();
    println!("phase1_probe.deno_core=ready");
    println!("phase1_probe.process=ready");
    drop_deno(javascript);
}

fn run_servo_only() {
    let servo = construct_servo();
    println!("phase1_probe.servo=ready");
    println!("phase1_probe.process=ready");
    drop_servo(servo);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).ok_or(
        "missing required mode: deno-only | servo-only | deno-then-servo | servo-then-deno | stress",
    )?;

    match mode.as_str() {
        "deno-only" => run_deno_only(),
        "deno-plain" => {
            let javascript = construct_deno_plain();
            println!("phase1_probe.deno_core=ready");
            println!("phase1_probe.process=ready");
            drop_deno(javascript);
        }
        "servo-only" => run_servo_only(),
        "deno-then-servo" => {
            let javascript = construct_deno();
            println!("phase1_probe.deno_core=ready");
            let servo = construct_servo();
            println!("phase1_probe.servo=ready");
            println!("phase1_probe.process=ready");
            drop_servo(servo);
            drop_deno(javascript);
        }
        "servo-then-deno" => {
            let servo = construct_servo();
            println!("phase1_probe.servo=ready");
            let javascript = construct_deno();
            println!("phase1_probe.deno_core=ready");
            println!("phase1_probe.process=ready");
            drop_deno(javascript);
            drop_servo(servo);
        }
        "stress" => {
            // Servo process-global Opts panics on a second Servo in the same
            // process. Repeat V8 Drop/regex/Intl here; Servo is constructed once.
            // Cross-order coverage is deno-then-servo / servo-then-deno.
            for round in 1..=3 {
                println!("phase1_probe.stress.round={round}");
                run_deno_only();
            }
            run_servo_only();
            println!("phase1_probe.stress=ok");
        }
        other => {
            return Err(format!(
                "unknown mode {other:?}; expected deno-only | servo-only | deno-then-servo | servo-then-deno | stress"
            )
            .into());
        }
    }

    Ok(())
}

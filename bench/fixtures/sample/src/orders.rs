// Order-handling module — provides the symbols the bench corpus queries
// (ProcessOrder, add_item, total).

use crate::ProcessOrder;

pub fn build_default_order() -> ProcessOrder {
    let mut o = ProcessOrder::new(42);
    o.add_item("widget");
    o.add_item("gadget");
    o
}

pub fn summary(o: &ProcessOrder) -> String {
    format!("order {}: {} items", o.id, o.total())
}

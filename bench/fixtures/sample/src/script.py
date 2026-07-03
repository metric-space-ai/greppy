"""Python file inside the Rust fixture. Exercises the
Language::Unsupported path: the file is discovered but not parsed
for graph nodes."""

def process_payment(amount, currency="EUR"):
    if amount <= 0:
        raise ValueError("amount must be positive")
    return f"charged {amount} {currency}"


class PaymentRetry:
    def __init__(self, max_attempts=3):
        self.attempts = 0
        self.max_attempts = max_attempts

    def try_again(self):
        if self.attempts >= self.max_attempts:
            return False
        self.attempts += 1
        return True

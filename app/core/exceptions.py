class UnsafeQueryError(Exception):
    def __init__(self, question: str) -> None:
        self.question = question
        super().__init__("Unsafe medical query detected")

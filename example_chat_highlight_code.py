from typing import Optional
import fire
from llama import Llama, Dialog
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.lexers import PythonLexer, MarkdownLexer, YamlLexer, JsonLexer
from pygments.formatters import TerminalFormatter


class InteractiveLlamaChat:
    def __init__(
        self,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
    ):
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len or max_seq_len
        print("Llama model loaded. You can start chatting!")

    def detect_format_and_highlight(self, text: str) -> str:
        # Simple format detection (this can be expanded with more sophisticated heuristics)
        if text.strip().startswith(("class ", "def ", "import ", "@")):
            lexer = PythonLexer()
        elif text.strip().startswith(("---", "#")):
            lexer = YamlLexer()
        elif text.strip().startswith(("{", "[")):
            lexer = JsonLexer()
        else:
            # Default to Markdown as it's more permissive
            lexer = MarkdownLexer()

        return highlight(text, lexer, TerminalFormatter())

    def chat(self):
        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() == "quit":
                    print("Exiting chat.")
                    break
                dialog = [{"role": "user", "content": user_input}]
                result = self.generator.chat_completion(
                    [dialog],
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                response = result[0]["generation"]["content"]
                highlighted_response = self.detect_format_and_highlight(response)
                print(f"Llama: {highlighted_response}")
        except KeyboardInterrupt:
            print("\nChat interrupted by user.")


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    chatbot = InteractiveLlamaChat(
        ckpt_dir,
        tokenizer_path,
        temperature,
        top_p,
        max_seq_len,
        max_batch_size,
        max_gen_len,
    )
    chatbot.chat()


if __name__ == "__main__":
    fire.Fire(main)

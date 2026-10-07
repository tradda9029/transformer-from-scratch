text = "hello world! hello transformer!"

chars = sorted(set(text))

vocab_size = len(chars)

stoi = {ch: i for i,ch in enumerate(chars)}#string to integer
itos = {i: ch for i,ch in enumerate(chars)} #integer to string

def encode(s: str) -> list[int]:
    return [stoi[ch] for ch in s]

def decode(ids: list[int]) -> str:
    decode_chars = [itos[i] for i in ids]
    return "".join(decode_chars)

if __name__ == "__main__":

    print(chars)
    print(vocab_size)

    print(encode("hello"))
    print(decode(encode("hello")))

    assert decode(encode(text)) == text 

    assert len(stoi) == vocab_size
    assert len(itos) == vocab_size

    assert isinstance(encode("hello"), list)
    assert all(isinstance(i, int) for i in encode("hello"))

    assert decode(encode(text)) == text

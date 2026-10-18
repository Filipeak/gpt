#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// GPT-2 byte-level BPE tokenizer, matching tiktoken's "gpt2" encoding.
// The vocab file is written by `python data/tokenizer.py export-vocab <file>`.
class Tokenizer
{
public:
    Tokenizer() = default;
    Tokenizer(const Tokenizer &) = delete; // ranks_ points into bytes_
    Tokenizer &operator=(const Tokenizer &) = delete;

    bool load(const char *vocab_path);

    // Encodes UTF-8 text. Special tokens like <|endoftext|> are treated as plain text (tiktoken's encode_ordinary).
    std::vector<int> encode(std::string_view text) const;

    // Raw bytes of one token. A token can end in the middle of a multi-byte UTF-8 character.
    std::string_view decode(int token) const;

    int vocab_size() const { return offsets_.empty() ? 0 : (int)offsets_.size() - 1; }
    int eot_token() const { return eot_token_; }

private:
    void encode_piece(std::string_view piece, std::vector<int> &tokens) const;
    int rank(std::string_view bytes) const;

    std::string bytes_;                                // every token's bytes, back to back
    std::vector<uint32_t> offsets_;                    // token i is bytes_[offsets_[i], offsets_[i + 1])
    std::unordered_map<std::string_view, int> ranks_;  // token bytes -> id; a lower id is merged first
    int eot_token_ = -1;
};

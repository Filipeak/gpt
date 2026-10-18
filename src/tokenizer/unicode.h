#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// Character classes of GPT-2's pre-tokenization regex.
enum class CharClass
{
    Letter,     // \p{L}
    Number,     // \p{N}
    Whitespace, // \s (Unicode White_Space)
    Other,      // everything else
};

// Decodes the UTF-8 character starting at text[pos] and returns its length in bytes.
// An invalid byte decodes on its own as U+FFFD.
size_t utf8_decode(std::string_view text, size_t pos, uint32_t *codepoint);

// Length of the longest prefix of bytes that does not end in the middle of a UTF-8 character.
size_t utf8_complete_length(std::string_view bytes);

CharClass char_class(uint32_t codepoint);

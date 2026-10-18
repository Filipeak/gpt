#include "unicode.h"
#include "unicode_ranges.h"

size_t utf8_decode(std::string_view text, size_t pos, uint32_t *codepoint)
{
    const unsigned char *s = (const unsigned char *)text.data() + pos;
    const size_t available = text.size() - pos;
    const unsigned char lead = s[0];

    *codepoint = 0xFFFD;

    if (lead < 0x80)
    {
        *codepoint = lead;
        return 1;
    }

    size_t len;
    uint32_t cp;
    uint32_t min_cp; // smaller values are overlong encodings

    if ((lead & 0xE0) == 0xC0)
    {
        len = 2, cp = lead & 0x1F, min_cp = 0x80;
    }
    else if ((lead & 0xF0) == 0xE0)
    {
        len = 3, cp = lead & 0x0F, min_cp = 0x800;
    }
    else if ((lead & 0xF8) == 0xF0)
    {
        len = 4, cp = lead & 0x07, min_cp = 0x10000;
    }
    else
    {
        return 1;
    }

    if (len > available)
    {
        return 1;
    }

    for (size_t i = 1; i < len; ++i)
    {
        if ((s[i] & 0xC0) != 0x80)
        {
            return 1;
        }

        cp = (cp << 6) | (s[i] & 0x3F);
    }

    if (cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
    {
        return 1;
    }

    *codepoint = cp;
    return len;
}

size_t utf8_complete_length(std::string_view bytes)
{
    const size_t n = bytes.size();

    // Walk back over continuation bytes to the lead byte of the last character. Only a
    // character with 1-3 of its bytes present can be incomplete.
    for (size_t back = 1; back <= 3 && back <= n; ++back)
    {
        const unsigned char c = bytes[n - back];

        if ((c & 0xC0) == 0x80)
        {
            continue;
        }

        size_t needed = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return needed > back ? n - back : n;
    }

    return n;
}

static bool in_ranges(const uint32_t (*ranges)[2], size_t count, uint32_t cp)
{
    size_t lo = 0;
    size_t hi = count;

    while (lo < hi)
    {
        size_t mid = (lo + hi) / 2;

        if (cp < ranges[mid][0])
            hi = mid;
        else if (cp > ranges[mid][1])
            lo = mid + 1;
        else
            return true;
    }

    return false;
}

// Unicode White_Space property, which is what \s matches in tiktoken's regex engine.
static bool is_whitespace(uint32_t cp)
{
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

CharClass char_class(uint32_t cp)
{
    if (is_whitespace(cp))
        return CharClass::Whitespace;
    if (in_ranges(LETTER_RANGES, sizeof(LETTER_RANGES) / sizeof(LETTER_RANGES[0]), cp))
        return CharClass::Letter;
    if (in_ranges(NUMBER_RANGES, sizeof(NUMBER_RANGES) / sizeof(NUMBER_RANGES[0]), cp))
        return CharClass::Number;

    return CharClass::Other;
}

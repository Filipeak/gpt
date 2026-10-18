#include "tokenizer.h"
#include "unicode.h"
#include "logger.h"
#include <climits>
#include <cstdio>
#include <cstring>

static const char VOCAB_MAGIC[4] = {'B', 'P', 'E', 'V'};
static const int NO_RANK = INT_MAX;

bool Tokenizer::load(const char *vocab_path)
{
    FILE *fp = fopen(vocab_path, "rb");

    if (!fp)
    {
        LOG_ERROR("Could not open vocab file %s for reading.", vocab_path);
        return false;
    }

    // Layout written by data/tokenizer.py: magic, u32 vocab_size, u32 eot_token, then per token a u8 length and its bytes.
    char magic[4];
    uint32_t vocab_size = 0;
    uint32_t eot_token = 0;

    bool ok = fread(magic, 1, 4, fp) == 4 && memcmp(magic, VOCAB_MAGIC, 4) == 0 &&
              fread(&vocab_size, sizeof(uint32_t), 1, fp) == 1 &&
              fread(&eot_token, sizeof(uint32_t), 1, fp) == 1 &&
              eot_token < vocab_size;

    bytes_.clear();
    offsets_.assign(1, 0);
    ranks_.clear();

    for (uint32_t i = 0; ok && i < vocab_size; ++i)
    {
        unsigned char len = 0;
        char buf[256];

        ok = fread(&len, 1, 1, fp) == 1 && len > 0 && fread(buf, 1, len, fp) == len;
        bytes_.append(buf, len);
        offsets_.push_back((uint32_t)bytes_.size());
    }

    fclose(fp);

    if (ok)
    {
        // bytes_ is final now, so views into it stay valid.
        ranks_.reserve(vocab_size);

        for (uint32_t i = 0; i < vocab_size; ++i)
        {
            if (i != eot_token)
            {
                ranks_.emplace(decode(i), (int)i);
            }
        }

        // BPE starts from single bytes, so every byte needs a token.
        for (int b = 0; ok && b < 256; ++b)
        {
            char byte = (char)b;
            ok = rank(std::string_view(&byte, 1)) != NO_RANK;
        }
    }

    if (!ok)
    {
        LOG_ERROR("Malformed vocab file %s.", vocab_path);
        bytes_.clear();
        offsets_.clear();
        ranks_.clear();
        return false;
    }

    eot_token_ = (int)eot_token;

    LOG_INFO("Loaded vocab from %s (%u tokens).", vocab_path, vocab_size);
    return true;
}

static CharClass class_at(std::string_view text, size_t pos, size_t *len)
{
    uint32_t cp;
    *len = utf8_decode(text, pos, &cp);
    return char_class(cp);
}

// Byte length of the pre-tokenization piece starting at text[start]. Follows GPT-2's regex
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// trying the alternatives in order. BPE never merges across piece boundaries.
static size_t next_piece_length(std::string_view text, size_t start)
{
    const size_t n = text.size();
    size_t len = 0;

    // Contractions: lowercase, ASCII apostrophe only.
    if (text[start] == '\'' && start + 1 < n)
    {
        char a = text[start + 1];

        if (a == 's' || a == 'd' || a == 'm' || a == 't')
        {
            return 2;
        }

        if (start + 2 < n)
        {
            char b = text[start + 2];

            if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e'))
            {
                return 3;
            }
        }
    }

    // An optional space, then a run of letters, of numbers, or of other symbols.
    size_t pos = start + (text[start] == ' ' ? 1 : 0);

    if (pos < n)
    {
        CharClass cls = class_at(text, pos, &len);

        if (cls != CharClass::Whitespace)
        {
            pos += len;

            while (pos < n && class_at(text, pos, &len) == cls)
            {
                pos += len;
            }

            return pos - start;
        }
    }

    // A whitespace run. If it is followed by a non-space and is longer than one character,
    // its last character is left to start the next piece, so "a  b" splits as "a", " ", " b".
    size_t last = start;
    pos = start;

    while (pos < n && class_at(text, pos, &len) == CharClass::Whitespace)
    {
        last = pos;
        pos += len;
    }

    if (pos == n || last == start)
    {
        return pos - start;
    }

    return last - start;
}

std::vector<int> Tokenizer::encode(std::string_view text) const
{
    std::vector<int> tokens;

    for (size_t pos = 0; pos < text.size();)
    {
        size_t len = next_piece_length(text, pos);
        encode_piece(text.substr(pos, len), tokens);
        pos += len;
    }

    return tokens;
}

std::string_view Tokenizer::decode(int token) const
{
    if (token < 0 || token >= vocab_size())
    {
        return {};
    }

    return std::string_view(bytes_).substr(offsets_[token], offsets_[token + 1] - offsets_[token]);
}

int Tokenizer::rank(std::string_view bytes) const
{
    auto it = ranks_.find(bytes);
    return it == ranks_.end() ? NO_RANK : it->second;
}

// Same algorithm as tiktoken's byte_pair_encode.
void Tokenizer::encode_piece(std::string_view piece, std::vector<int> &tokens) const
{
    int whole = rank(piece);

    if (whole != NO_RANK)
    {
        tokens.push_back(whole);
        return;
    }

    // Start from single bytes and keep merging the adjacent pair with the lowest rank (the leftmost
    // on a tie). Part i is piece[parts[i].start, parts[i + 1].start); the last entry only marks the end.
    // pair_rank is the rank of part i joined with part i + 1.
    struct Part
    {
        size_t start;
        int pair_rank;
    };

    std::vector<Part> parts;
    parts.reserve(piece.size() + 1);

    for (size_t i = 0; i + 1 < piece.size(); ++i)
    {
        parts.push_back({i, rank(piece.substr(i, 2))});
    }

    parts.push_back({piece.size() - 1, NO_RANK});
    parts.push_back({piece.size(), NO_RANK});

    // Rank of parts i, i + 1 and i + 2 joined; called before part i + 1 is merged into part i.
    auto joined_rank = [&](size_t i)
    {
        return i + 3 < parts.size() ? rank(piece.substr(parts[i].start, parts[i + 3].start - parts[i].start)) : NO_RANK;
    };

    while (true)
    {
        size_t best = 0;
        int best_rank = NO_RANK;

        for (size_t i = 0; i + 1 < parts.size(); ++i)
        {
            if (parts[i].pair_rank < best_rank)
            {
                best_rank = parts[i].pair_rank;
                best = i;
            }
        }

        if (best_rank == NO_RANK)
        {
            break;
        }

        if (best > 0)
        {
            parts[best - 1].pair_rank = joined_rank(best - 1);
        }

        parts[best].pair_rank = joined_rank(best);
        parts.erase(parts.begin() + best + 1);
    }

    for (size_t i = 0; i + 1 < parts.size(); ++i)
    {
        tokens.push_back(rank(piece.substr(parts[i].start, parts[i + 1].start - parts[i].start)));
    }
}

#include "tokenizer.h"
#include "unicode.h"
#include "logger.h"
#include "benchmark.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char *VOCAB_FILE = "./data/gpt2_vocab.bin";
static const char *CASES_FILE = "./data/tokenizer_cases.bin";
static const int MAX_REPORTED_FAILURES = 20;

struct TestCase
{
    std::string name;
    std::string text;
    std::vector<int> tokens;
};

static int passed = 0;
static int failed = 0;

static void check(bool ok, const char *name)
{
    if (ok)
    {
        ++passed;
        LOG_INFO("[ PASS ] %s", name);
    }
    else
    {
        ++failed;
        LOG_ERROR("[ FAIL ] %s", name);
    }
}

// Printable ASCII with \n, \t and \xNN escapes.
static std::string escape(std::string_view bytes)
{
    std::string out;
    char buf[8];

    for (unsigned char c : bytes)
    {
        if (c == '\n')
            out += "\\n";
        else if (c == '\t')
            out += "\\t";
        else if (c >= 0x20 && c < 0x7F && c != '\\')
            out += (char)c;
        else
        {
            snprintf(buf, sizeof(buf), "\\x%02X", c);
            out += buf;
        }
    }

    return out;
}

static bool read_file(const char *path, std::string *out)
{
    FILE *fp = fopen(path, "rb");

    if (!fp)
    {
        LOG_ERROR("Could not open %s for reading.", path);
        return false;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    out->resize(size);
    size_t count = fread(out->data(), 1, size, fp);
    fclose(fp);

    return count == (size_t)size;
}

// Format written by tools/prep_tokenizer.py.
static bool load_cases(const char *path, std::vector<TestCase> *cases)
{
    std::string data;

    if (!read_file(path, &data))
    {
        return false;
    }

    size_t pos = 0;
    bool ok = true;

    auto read_u32 = [&]() -> uint32_t
    {
        uint32_t v = 0;
        if (pos + 4 > data.size())
        {
            ok = false;
            return 0;
        }
        memcpy(&v, data.data() + pos, 4);
        pos += 4;
        return v;
    };

    auto read_bytes = [&](uint32_t n) -> std::string
    {
        if (pos + n > data.size())
        {
            ok = false;
            return {};
        }
        std::string s = data.substr(pos, n);
        pos += n;
        return s;
    };

    uint32_t count = read_u32();

    for (uint32_t i = 0; i < count && ok; ++i)
    {
        TestCase c;
        c.name = read_bytes(read_u32());
        c.text = read_bytes(read_u32());

        uint32_t n_tokens = read_u32();
        c.tokens.resize(n_tokens);

        for (uint32_t t = 0; t < n_tokens && ok; ++t)
        {
            c.tokens[t] = (int)read_u32();
        }

        cases->push_back(std::move(c));
    }

    if (!ok || pos != data.size())
    {
        LOG_ERROR("Malformed test case file %s.", path);
        return false;
    }

    return true;
}

static void report_mismatch(const Tokenizer &tok, const TestCase &c, const std::vector<int> &got)
{
    size_t i = 0;
    while (i < got.size() && i < c.tokens.size() && got[i] == c.tokens[i])
    {
        ++i;
    }

    LOG_ERROR("         %s: %zu tokens expected, %zu got, first difference at token %zu", c.name.c_str(), c.tokens.size(), got.size(), i);

    size_t from = i >= 2 ? i - 2 : 0;
    std::string expected_str, got_str;

    for (size_t k = from; k < from + 6; ++k)
    {
        if (k < c.tokens.size())
            expected_str += std::to_string(c.tokens[k]) + "'" + escape(tok.decode(c.tokens[k])) + "' ";
        if (k < got.size())
            got_str += std::to_string(got[k]) + "'" + escape(tok.decode(got[k])) + "' ";
    }

    LOG_ERROR("         expected: %s", expected_str.c_str());
    LOG_ERROR("         got:      %s", got_str.c_str());
}

static void test_utf8_complete_length()
{
    LOG_INFO("-- utf8_complete_length --");

    check(utf8_complete_length("") == 0, "empty");
    check(utf8_complete_length("abc") == 3, "ascii");
    check(utf8_complete_length("a\xC3") == 1, "2-byte char missing 1 byte");
    check(utf8_complete_length("a\xC3\xA9") == 3, "2-byte char complete");
    check(utf8_complete_length("\xE2\x82") == 0, "3-byte char missing 1 byte");
    check(utf8_complete_length("x\xE2\x82\xAC") == 4, "3-byte char complete");
    check(utf8_complete_length("\xF0\x9F\x91") == 0, "4-byte char missing 1 byte");
    check(utf8_complete_length("\xF0\x9F\x91\x8B") == 4, "4-byte char complete");
    check(utf8_complete_length("a\x80") == 2, "stray continuation byte is not held back");
    check(utf8_complete_length("a\xFF") == 2, "invalid byte is not held back");
}

static void test_vocab(const Tokenizer &tok)
{
    LOG_INFO("-- vocab --");

    check(tok.vocab_size() == 50257, "vocab size is 50257");
    check(tok.eot_token() == 50256, "<|endoftext|> is token 50256");
    check(tok.decode(50256) == "<|endoftext|>", "decode <|endoftext|>");
    check(tok.decode(15496) == "Hello", "decode 15496 -> 'Hello'");
    check(tok.decode(-1).empty() && tok.decode(50257).empty(), "out-of-range ids decode to nothing");
}

static void test_cases(const Tokenizer &tok, const std::vector<TestCase> &cases)
{
    LOG_INFO("-- encode / decode vs tiktoken (%zu cases from %s) --", cases.size(), CASES_FILE);

    int reported = 0;
    int cases_failed = 0;
    size_t total_bytes = 0;
    double total_ms = 0.0;

    for (const TestCase &c : cases)
    {
        std::vector<int> got;
        BENCHMARK_SCOPE(Encode, {
            got = tok.encode(c.text);
        });
        total_bytes += c.text.size();
        total_ms += elapsed_Encode;

        std::string decoded;
        for (int t : c.tokens)
        {
            decoded += tok.decode(t);
        }

        bool encode_ok = got == c.tokens;
        bool decode_ok = decoded == c.text;

        if (encode_ok && decode_ok)
        {
            ++passed;
            continue;
        }

        ++failed;
        ++cases_failed;

        if (reported++ < MAX_REPORTED_FAILURES)
        {
            LOG_ERROR("[ FAIL ] %s (%s)", c.name.c_str(), !encode_ok ? "encode" : "decode");
            if (!encode_ok)
            {
                report_mismatch(tok, c, got);
            }
        }
    }

    LOG_INFO("%zu/%zu cases match tiktoken", cases.size() - cases_failed, cases.size());
    LOG_INFO("encoded %.2f MB in %.1f ms (%.1f MB/s)", total_bytes / 1e6, total_ms, total_bytes / 1e3 / (total_ms > 0.0 ? total_ms : 1.0));
}

int main()
{
    test_utf8_complete_length();

    Tokenizer tok;
    std::vector<TestCase> cases;

    if (!tok.load(VOCAB_FILE))
    {
        LOG_ERROR("Failed to load vocab. Run: python data/tokenizer.py export-vocab %s", VOCAB_FILE);
        return 1;
    }

    if (!load_cases(CASES_FILE, &cases))
    {
        LOG_ERROR("Failed to load test cases. Run: python tools/prep_tokenizer.py");
        return 1;
    }

    test_vocab(tok);
    test_cases(tok, cases);

    LOG_INFO("--- %d passed, %d failed ---", passed, failed);
    return failed == 0 ? 0 : 1;
}

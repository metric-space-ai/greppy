// Exact-token llama.cpp baseline for Greppy's inference performance contract.

#include "ggml-backend.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sched.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

using json = nlohmann::json;
using steady_clock = std::chrono::steady_clock;

constexpr const char * RAW_SCHEMA_VERSION = "greppy.inference-performance.raw.v1";

struct options {
    std::string model;
    std::string cases;
    std::string model_family;
    std::string workload;
    std::string device;
    int threads = 0;
    int samples = 5;
    int warmups = 1;
};

struct token_case {
    std::string id;
    std::vector<llama_token> input;
    std::vector<std::uint32_t> attention_mask;
};

struct model_deleter {
    void operator()(llama_model * value) const {
        if (value != nullptr) {
            llama_model_free(value);
        }
    }
};

struct context_deleter {
    void operator()(llama_context * value) const {
        if (value != nullptr) {
            llama_free(value);
        }
    }
};

using model_ptr = std::unique_ptr<llama_model, model_deleter>;
using context_ptr = std::unique_ptr<llama_context, context_deleter>;

void usage(const char * program) {
    std::cerr
        << "usage: " << program
        << " --model MODEL.gguf --cases NATIVE.jsonl"
           " --model-family qwen35_mtp|embeddinggemma"
           " --workload qwen_pp512|qwen_tg128|embedding_encoder"
           " --device cpu|metal|cuda --threads N [--samples N] [--warmups N]\n";
}

int positive_integer(const std::string & value, const std::string & name) {
    std::size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception &) {
        throw std::runtime_error(name + " must be a positive integer");
    }
    if (consumed != value.size() || parsed <= 0) {
        throw std::runtime_error(name + " must be a positive integer");
    }
    return parsed;
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--model") {
            result.model = value;
        } else if (argument == "--cases") {
            result.cases = value;
        } else if (argument == "--model-family") {
            result.model_family = value;
        } else if (argument == "--workload") {
            result.workload = value;
        } else if (argument == "--device") {
            result.device = value;
        } else if (argument == "--threads") {
            result.threads = positive_integer(value, "--threads");
        } else if (argument == "--samples") {
            result.samples = positive_integer(value, "--samples");
        } else if (argument == "--warmups") {
            result.warmups = positive_integer(value, "--warmups");
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (result.model.empty() || result.cases.empty() || result.threads == 0) {
        throw std::runtime_error("--model, --cases, and --threads are required");
    }
    if (result.model_family != "qwen35_mtp" && result.model_family != "embeddinggemma") {
        throw std::runtime_error("unsupported --model-family: " + result.model_family);
    }
    if (result.workload != "qwen_pp512" && result.workload != "qwen_tg128" &&
        result.workload != "embedding_encoder") {
        throw std::runtime_error("unsupported --workload: " + result.workload);
    }
    if ((result.workload == "embedding_encoder") != (result.model_family == "embeddinggemma")) {
        throw std::runtime_error("embedding_encoder requires embeddinggemma; Qwen workloads require qwen35_mtp");
    }
    if (result.device != "cpu" && result.device != "metal" && result.device != "cuda") {
        throw std::runtime_error("--device must be cpu, metal, or cuda");
    }
    return result;
}

std::vector<llama_token> parse_token_ids(const json & value, const std::string & label) {
    if (!value.is_array() || value.empty()) {
        throw std::runtime_error(label + " must be a non-empty token ID array");
    }
    std::vector<llama_token> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (!value[index].is_number_unsigned() && !value[index].is_number_integer()) {
            throw std::runtime_error(label + " contains a non-integer token ID");
        }
        const std::int64_t token = value[index].get<std::int64_t>();
        if (token < 0 || token > std::numeric_limits<llama_token>::max()) {
            throw std::runtime_error(label + " contains a token ID outside llama_token");
        }
        result.push_back(static_cast<llama_token>(token));
    }
    return result;
}

std::vector<token_case> read_cases(const options & opts) {
    std::ifstream input(opts.cases);
    if (!input) {
        throw std::runtime_error("cannot open token cases: " + opts.cases);
    }
    std::map<std::string, std::vector<llama_token>> unique;
    std::map<std::string, std::vector<std::uint32_t>> unique_masks;
    std::vector<std::string> order;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        json value;
        try {
            value = json::parse(line);
        } catch (const json::exception & error) {
            throw std::runtime_error(
                opts.cases + ":" + std::to_string(line_number) + ": invalid JSON: " + error.what());
        }
        if (!value.is_object() || !value.contains("case_id") || !value["case_id"].is_string() ||
            !value.contains("input_token_ids")) {
            throw std::runtime_error(
                opts.cases + ":" + std::to_string(line_number) +
                ": expected case_id and input_token_ids");
        }
        if (value.contains("model_family") && value["model_family"] != opts.model_family) {
            throw std::runtime_error("token case model_family differs from --model-family");
        }
        // The native Qwen producer emits PP512, TG128, and production-prompt
        // rows together. Select the requested workload from that provenance-
        // bound file so both llama.cpp runs consume the exact native tokens.
        if (value.contains("workload") && value["workload"] != opts.workload) {
            continue;
        }
        const std::string id = value["case_id"].get<std::string>();
        if (id.empty()) {
            throw std::runtime_error("token case has an empty case_id");
        }
        auto tokens = parse_token_ids(value["input_token_ids"], "input_token_ids for " + id);
        std::vector<std::uint32_t> attention_mask;
        if (value.contains("attention_mask") && value["attention_mask"].is_array()) {
            for (const auto & mask_value : value["attention_mask"]) {
                if (!mask_value.is_number_integer()) {
                    throw std::runtime_error(id + ": attention_mask contains a non-integer");
                }
                const int parsed = mask_value.get<int>();
                if (parsed != 0 && parsed != 1) {
                    throw std::runtime_error(id + ": attention_mask values must be 0 or 1");
                }
                attention_mask.push_back(static_cast<std::uint32_t>(parsed));
            }
        }
        if (opts.workload == "embedding_encoder" && attention_mask.size() != tokens.size()) {
            throw std::runtime_error(id + ": embedding attention_mask must match input token count");
        }
        const auto existing = unique.find(id);
        if (existing == unique.end()) {
            unique.emplace(id, tokens);
            unique_masks.emplace(id, attention_mask);
            order.push_back(id);
        } else if (existing->second != tokens || unique_masks.at(id) != attention_mask) {
            throw std::runtime_error(id + ": input token IDs or attention mask changed between raw samples");
        }
    }
    if (order.empty()) {
        throw std::runtime_error(
            "token case JSONL contains no rows for requested workload " + opts.workload);
    }
    std::vector<token_case> result;
    result.reserve(order.size());
    for (const auto & id : order) {
        result.push_back({id, unique.at(id), unique_masks.at(id)});
    }
    return result;
}

void silent_log(enum ggml_log_level, const char *, void *) {}

void configure_cpu_contract(const options & opts) {
#if defined(__APPLE__)
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0) {
        throw std::runtime_error("cannot select Apple performance QoS");
    }
    int performance_cores = 0;
    std::size_t size = sizeof(performance_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &performance_cores, &size, nullptr, 0) != 0 ||
        performance_cores <= 0) {
        throw std::runtime_error("cannot determine Apple performance-core count");
    }
    if (opts.threads != performance_cores) {
        throw std::runtime_error(
            "--threads must equal the Apple performance-core count " +
            std::to_string(performance_cores));
    }
#elif defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        throw std::runtime_error("cannot read Linux CPU affinity");
    }
    int allowed_count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            ++allowed_count;
        }
    }
    if (allowed_count != opts.threads) {
        throw std::runtime_error(
            "Linux process affinity must expose exactly --threads CPUs; use one logical CPU per P-core");
    }
#elif defined(_WIN32)
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0) {
        throw std::runtime_error("cannot read Windows process affinity");
    }
    int allowed_count = 0;
    while (process_mask != 0) {
        allowed_count += static_cast<int>(process_mask & 1U);
        process_mask >>= 1U;
    }
    if (allowed_count != opts.threads) {
        throw std::runtime_error(
            "Windows process affinity must expose exactly --threads logical CPUs");
    }
#else
    (void) opts;
#endif
}

std::vector<ggml_backend_dev_t> select_devices(const std::string & requested) {
    std::vector<ggml_backend_dev_t> gpus;
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        auto * device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(device);
        }
    }
    if (requested == "cpu") {
        return {nullptr};
    }
    if (gpus.size() != 1) {
        throw std::runtime_error(
            "GPU contract requires exactly one enumerated GPU, found " + std::to_string(gpus.size()));
    }
    const std::string name = ggml_backend_dev_name(gpus.front());
    std::string normalized = name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    if (requested == "metal" && normalized.find("metal") == std::string::npos &&
        normalized.find("mtl") == std::string::npos) {
        throw std::runtime_error("the sole GPU is not a Metal device: " + name);
    }
    if (requested == "cuda" && normalized.find("cuda") == std::string::npos) {
        // NVIDIA device names often omit the backend, so accept them when CUDA visibility is explicit.
        const char * visible = std::getenv("CUDA_VISIBLE_DEVICES");
        if (visible == nullptr || std::string(visible).empty() || std::string(visible).find(',') != std::string::npos) {
            throw std::runtime_error(
                "CUDA requires CUDA_VISIBLE_DEVICES to contain exactly one device ID");
        }
    }
    gpus.push_back(nullptr);
    return gpus;
}

context_ptr make_context(llama_model * model, const token_case & item, const options & opts) {
    llama_context_params params = llama_context_default_params();
    const std::size_t extra = opts.workload == "qwen_tg128" ? 128U : 0U;
    const std::size_t required = item.input.size() + extra;
    if (required > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(item.id + ": context length exceeds u32");
    }
    params.n_ctx = static_cast<std::uint32_t>(std::max<std::size_t>(required, 512U));
    params.n_batch = static_cast<std::uint32_t>(std::max<std::size_t>(item.input.size(), 1U));
    params.n_ubatch = params.n_batch;
    params.n_seq_max = 1;
    params.n_threads = opts.threads;
    params.n_threads_batch = opts.threads;
    params.no_perf = true;
    params.embeddings = opts.workload == "embedding_encoder";
    if (opts.device == "cpu") {
        params.offload_kqv = false;
        params.op_offload = false;
    }
    if (params.embeddings) {
        params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    }
    context_ptr context(llama_init_from_model(model, params));
    if (!context) {
        throw std::runtime_error(item.id + ": cannot create llama context");
    }
    llama_set_n_threads(context.get(), opts.threads, opts.threads);
    if (params.embeddings) {
        llama_set_embeddings(context.get(), true);
        llama_set_causal_attn(context.get(), false);
    }
    return context;
}

void clear_context(llama_context * context) {
    llama_memory_clear(llama_get_memory(context), false);
}

std::uint64_t elapsed_ns(steady_clock::time_point started) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(steady_clock::now() - started);
    if (elapsed.count() <= 0) {
        throw std::runtime_error("timer returned a non-positive duration");
    }
    return static_cast<std::uint64_t>(elapsed.count());
}

std::uint64_t run_prefill(llama_context * context, std::vector<llama_token> & input) {
    clear_context(context);
    const auto started = steady_clock::now();
    if (llama_decode(context, llama_batch_get_one(input.data(), static_cast<std::int32_t>(input.size()))) != 0) {
        throw std::runtime_error("llama_decode failed during PP512");
    }
    llama_synchronize(context);
    return elapsed_ns(started);
}

llama_token greedy_token(llama_context * context, int vocab_size) {
    const float * logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) {
        throw std::runtime_error("llama.cpp did not expose logits for greedy TG128");
    }
    return static_cast<llama_token>(
        std::max_element(logits, logits + vocab_size) - logits);
}

std::pair<std::uint64_t, std::vector<llama_token>> run_generation(
    llama_context * context,
    std::vector<llama_token> & input,
    int vocab_size) {
    clear_context(context);
    if (llama_decode(context, llama_batch_get_one(input.data(), static_cast<std::int32_t>(input.size()))) != 0) {
        throw std::runtime_error("llama_decode failed during TG128 prefill");
    }
    llama_synchronize(context);
    std::vector<llama_token> output;
    output.reserve(128);
    const auto started = steady_clock::now();
    for (std::size_t index = 0; index < 128; ++index) {
        const llama_token token = greedy_token(context, vocab_size);
        output.push_back(token);
        if (index + 1 < 128) {
            if (llama_decode(context, llama_batch_get_one(&output.back(), 1)) != 0) {
                throw std::runtime_error("llama_decode failed during TG128 generation");
            }
        }
    }
    return {elapsed_ns(started), std::move(output)};
}

std::uint64_t run_encoder(
    llama_context * context,
    std::vector<llama_token> & input,
    bool model_has_encoder) {
    clear_context(context);
    const auto started = steady_clock::now();
    const auto batch = llama_batch_get_one(input.data(), static_cast<std::int32_t>(input.size()));
    const int result = model_has_encoder ? llama_encode(context, batch) : llama_decode(context, batch);
    if (result != 0) {
        throw std::runtime_error("llama.cpp failed during EmbeddingGemma non-causal encoder forward");
    }
    llama_synchronize(context);
    const float * embedding = llama_get_embeddings_seq(context, 0);
    if (embedding == nullptr) {
        throw std::runtime_error("llama.cpp did not expose the pooled encoder embedding");
    }
    volatile float sink = embedding[0];
    (void) sink;
    return elapsed_ns(started);
}

const char * semantics(const std::string & workload) {
    if (workload == "qwen_pp512") {
        return "qwen_target_prefill_exact_512_v1";
    }
    if (workload == "qwen_tg128") {
        return "qwen_greedy_generation_exact_128_v1";
    }
    return "embeddinggemma_encoder_forward_v1";
}

const char * generation_path(const std::string & workload) {
    if (workload == "qwen_pp512") {
        return "target_prefill";
    }
    if (workload == "qwen_tg128") {
        return "target_greedy_reference";
    }
    return "encoder";
}

void emit_sample(
    const options & opts,
    const token_case & item,
    int sample_index,
    std::uint64_t duration,
    const std::vector<llama_token> & output) {
    json value = {
        {"schema_version", RAW_SCHEMA_VERSION},
        {"model_family", opts.model_family},
        {"workload", opts.workload},
        {"semantics", semantics(opts.workload)},
        {"generation_path", generation_path(opts.workload)},
        {"case_id", item.id},
        {"sample_index", sample_index},
        {"elapsed_ns", duration},
        {"input_token_ids", item.input},
        {"attention_mask", item.attention_mask},
        {"output_token_ids", output},
        {"output_limit", opts.workload == "qwen_tg128" ? 128 : 0},
        {"threads", opts.threads},
        {"device", opts.device},
    };
    std::cout << value.dump() << '\n';
}

void validate_case(const token_case & item, const options & opts, int vocab_size) {
    if (opts.workload == "qwen_pp512" && item.input.size() != 512) {
        throw std::runtime_error(item.id + ": PP512 requires exactly 512 input token IDs");
    }
    for (const llama_token token : item.input) {
        if (token < 0 || token >= vocab_size) {
            throw std::runtime_error(item.id + ": input contains a token ID outside the model vocabulary");
        }
    }
    if (opts.workload == "embedding_encoder" &&
        std::any_of(item.attention_mask.begin(), item.attention_mask.end(), [](std::uint32_t value) {
            return value != 1;
        })) {
        throw std::runtime_error(
            item.id + ": llama.cpp exact-token encoder baseline requires an unpadded all-ones mask");
    }
}

void run(const options & opts) {
    auto cases = read_cases(opts);
    configure_cpu_contract(opts);
    llama_log_set(silent_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    auto selected_devices = select_devices(opts.device);
    llama_model_params model_params = llama_model_default_params();
    if (opts.device == "cpu") {
        model_params.n_gpu_layers = 0;
        model_params.devices = selected_devices.data();
    } else {
        model_params.n_gpu_layers = std::numeric_limits<std::int32_t>::max();
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.main_gpu = 0;
        model_params.devices = selected_devices.data();
    }
    model_ptr model(llama_model_load_from_file(opts.model.c_str(), model_params));
    if (!model) {
        throw std::runtime_error("cannot load llama.cpp model");
    }
    const bool model_has_encoder = llama_model_has_encoder(model.get());
    if (opts.workload != "embedding_encoder" && model_has_encoder) {
        throw std::runtime_error("Qwen workload cannot use an encoder-only model");
    }
    const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    if (vocab_size <= 0) {
        throw std::runtime_error("model vocabulary is empty");
    }

    for (auto & item : cases) {
        validate_case(item, opts, vocab_size);
        auto context = make_context(model.get(), item, opts);
        for (int warmup = 0; warmup < opts.warmups; ++warmup) {
            if (opts.workload == "qwen_pp512") {
                (void) run_prefill(context.get(), item.input);
            } else if (opts.workload == "qwen_tg128") {
                (void) run_generation(context.get(), item.input, vocab_size);
            } else {
                (void) run_encoder(context.get(), item.input, model_has_encoder);
            }
        }
        for (int sample = 0; sample < opts.samples; ++sample) {
            std::uint64_t duration = 0;
            std::vector<llama_token> output;
            if (opts.workload == "qwen_pp512") {
                duration = run_prefill(context.get(), item.input);
            } else if (opts.workload == "qwen_tg128") {
                auto result = run_generation(context.get(), item.input, vocab_size);
                duration = result.first;
                output = std::move(result.second);
            } else {
                duration = run_encoder(context.get(), item.input, model_has_encoder);
            }
            emit_sample(opts, item, sample, duration, output);
        }
    }
    model.reset();
    llama_backend_free();
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto opts = parse_options(argc, argv);
        run(opts);
        return 0;
    } catch (const std::exception & error) {
        usage(argv[0]);
        std::cerr << "llama contract failed: " << error.what() << '\n';
        return 1;
    }
}

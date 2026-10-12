#ifndef FMA_PROTOCOL_H
#define FMA_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FMA_MAGIC UINT32_C(0x31414d46) /* FMA1 on a little-endian wire. */
#define FMA_PROTOCOL_MAJOR 1u
#define FMA_PROTOCOL_MINOR 2u
#define FMA_WIRE_HEADER_SIZE 48u
#define FMA_MAX_PAYLOAD (8u * 1024u * 1024u)
#define FMA_MAX_FDS 4u
#define FMA_DEFAULT_SLOTS 4u
#define FMA_MAX_SLOTS 8u
#define FMA_MAX_DIMENSION 8192u
#define FMA_MAX_POOL_BYTES (UINT64_C(512) * 1024u * 1024u)
#define FMA_DIRECT_OUTPUT_SLOT UINT32_MAX

enum fma_message_type {
    FMA_MSG_HELLO = 1,
    FMA_MSG_HELLO_REPLY = 2,
    FMA_MSG_QUERY_CAPABILITIES = 3,
    FMA_MSG_CAPABILITIES = 4,
    FMA_MSG_CREATE_DECODER = 5,
    FMA_MSG_DECODER_READY = 6,
    FMA_MSG_FRAME_POOL = 7,
    FMA_MSG_QUEUE_PACKET = 8,
    FMA_MSG_PACKET_ACK = 9,
    FMA_MSG_FRAME_READY = 10,
    FMA_MSG_RELEASE_FRAME = 11,
    FMA_MSG_FLUSH = 12,
    FMA_MSG_FLUSHED = 13,
    FMA_MSG_DRAIN = 14,
    FMA_MSG_OUTPUT_EOS = 15,
    FMA_MSG_CLOSE = 16,
    FMA_MSG_ERROR = 17,
    FMA_MSG_POLL_OUTPUT = 18,
    FMA_MSG_POLL_DONE = 19,
};

enum fma_codec {
    FMA_CODEC_H264 = 1,
    FMA_CODEC_HEVC = 2,
    FMA_CODEC_VP8 = 3,
    FMA_CODEC_VP9 = 4,
    FMA_CODEC_AV1 = 5,
};

#define FMA_CODEC_BIT(codec) (UINT32_C(1) << ((codec) - 1u))

enum fma_pixel_format {
    FMA_PIXFMT_NV12 = 1,
};

enum fma_packet_flags {
    FMA_PACKET_CODEC_CONFIG = 1u << 0,
    FMA_PACKET_KEY_FRAME = 1u << 1,
};

enum fma_capability_flags {
    FMA_CAP_SHARED_FRAME_POOL = 1u << 0,
    FMA_CAP_CAN_FLUSH = 1u << 1,
    FMA_CAP_CAN_POLL = 1u << 2,
    FMA_CAP_DIRECT_OUTPUT = 1u << 3,
};

struct fma_message {
    uint16_t type;
    uint32_t flags;
    uint32_t payload_size;
    uint32_t fd_count;
    uint64_t request_id;
    uint64_t session_id;
    int64_t pts_us;
    uint8_t *payload;
    int fds[FMA_MAX_FDS];
};

struct fma_capabilities {
    uint32_t decoder_mask;
    uint32_t pixel_format_mask;
    uint32_t flags;
    uint32_t max_width;
    uint32_t max_height;
};

struct fma_decoder_config {
    uint32_t codec;
    uint32_t width;
    uint32_t height;
    uint32_t slot_count;
};

struct fma_frame_pool {
    uint32_t pixel_format;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t slot_count;
    uint32_t slot_size;
};

struct fma_frame {
    uint32_t slot;
    uint32_t bytes_used;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixel_format;
};

void fma_message_init(struct fma_message *message, uint16_t type);
void fma_message_release(struct fma_message *message);

void fma_put_u16(uint8_t *dst, uint16_t value);
void fma_put_u32(uint8_t *dst, uint32_t value);
void fma_put_u64(uint8_t *dst, uint64_t value);
uint16_t fma_get_u16(const uint8_t *src);
uint32_t fma_get_u32(const uint8_t *src);
uint64_t fma_get_u64(const uint8_t *src);

int fma_encode_header(const struct fma_message *message,
                      uint8_t out[FMA_WIRE_HEADER_SIZE]);
int fma_decode_header(const uint8_t in[FMA_WIRE_HEADER_SIZE],
                      struct fma_message *message);

void fma_encode_capabilities(const struct fma_capabilities *caps, uint8_t out[20]);
int fma_decode_capabilities(const uint8_t *data, size_t size,
                            struct fma_capabilities *caps);
void fma_encode_decoder_config(const struct fma_decoder_config *config,
                               uint8_t out[16]);
int fma_decode_decoder_config(const uint8_t *data, size_t size,
                              struct fma_decoder_config *config);
void fma_encode_frame_pool(const struct fma_frame_pool *pool, uint8_t out[24]);
int fma_decode_frame_pool(const uint8_t *data, size_t size,
                          struct fma_frame_pool *pool);
void fma_encode_frame(const struct fma_frame *frame, uint8_t out[24]);
int fma_decode_frame(const uint8_t *data, size_t size, struct fma_frame *frame);

const char *fma_codec_name(uint32_t codec);
const char *fma_codec_mime(uint32_t codec);

#ifdef __cplusplus
}
#endif

#endif

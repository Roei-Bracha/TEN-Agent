# gemini_v2v_python

An extension for integrating Gemini's Next Generation of **Multimodal** AI into your application, providing configurable AI-driven features such as conversational agents, task automation, and tool integration.

## Features

- Gemini **Multimodal** Integration: Leverage Gemini **Multimodal** models for voice-to-voice as well as text processing.
- Configurable: Easily customize API keys, model settings, prompts, temperature, etc.
- Async Queue Processing: Supports real-time message processing with task cancellation and prioritization.
- **Affective Dialog**: Enables Gemini to adapt its response style to match the input expression and tone.
- **Proactive Audio**: Allows Gemini to intelligently decide when not to respond if content is not relevant.
- **Session Resumption**: Supports session resumption based on the [Gemini Live API documentation](https://ai.google.dev/gemini-api/docs/live#session-resumption).

## API

Refer to the `api` definition in [manifest.json] and default values in [property.json](property.json).

| **Property**               | **Type**   | **Description**                           |
|----------------------------|------------|-------------------------------------------|
| `api_key`                   | `string`   | API key for authenticating with Gemini    |
| `temperature`               | `float32`  | Sampling temperature, higher values mean more randomness |
| `model`                     | `string`   | Model identifier (e.g., GPT-4, Gemini-1)  |
| `max_tokens`                | `int32`    | Maximum number of tokens to generate      |
| `system_message`            | `string`   | Default system message to send to the model |
| `voice`                     | `string`   | Voice that Gemini model uses, such as `alloy`, `echo`, `shimmer`, etc. |
| `server_vad`                | `bool`     | Flag to enable or disable server VAD for Gemini |
| `language`                  | `string`   | Language that Gemini model responds in, such as `en-US`, `zh-CN`, etc. |
| `dump`                      | `bool`     | Flag to enable or disable audio dump for debugging purposes |
| `base_uri`                  | `string`   | Base URI for connecting to the Gemini service |
| `audio_out`                 | `bool`     | Flag to enable or disable audio output    |
| `input_transcript`          | `bool`     | Flag to enable input transcript processing |
| `sample_rate`               | `int32`    | Sample rate for audio processing          |
| `stream_id`                 | `int32`    | Stream ID for identifying audio streams   |
| `greeting`                  | `string`   | Greeting message for initial interaction  |

### Data Out

| **Name**       | **Property** | **Type**   | **Description**               |
|----------------|--------------|------------|-------------------------------|
| `text_data`    | `text`       | `string`   | Outgoing text data             |
| `append`       | `text`       | `string`   | Additional text appended to the output |

### Command Out

| **Name**       | **Description**                             |
|----------------|---------------------------------------------|
| `flush`        | Response after flushing the current state    |
| `tool_call`    | Invokes a tool with specific arguments       |

### Audio Frame In

| **Name**         | **Description**                           |
|------------------|-------------------------------------------|
| `pcm_frame`      | Audio frame input for voice processing    |

### Video Frame In

| **Name**         | **Description**                           |
|------------------|-------------------------------------------|
| `video_frame`    | Video frame input for processing          |

### Audio Frame Out

| **Name**         | **Description**                           |
|------------------|-------------------------------------------|
| `pcm_frame`      | Audio frame output after voice processing |

## Advanced Features

### Affective Dialog

This feature lets Gemini adapt its response style to the input expression and tone.

To use affective dialog, set `enable_affective_dialog` to `true` in the setup message:

```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    enable_affective_dialog=True
)
```

Note that affective dialog is currently only supported by the native audio output models.

### Proactive Audio

When this feature is enabled, Gemini can proactively decide not to respond if the content is not relevant.

To use it, configure the proactivity field in the setup message and set `proactive_audio` to `true`:

```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    proactivity={'proactive_audio': True}
)
```

Note that proactive audio is currently only supported by the native audio output models.

## Session Resumption

The extension now supports **session resumption** based on the [Gemini Live API documentation](https://ai.google.dev/gemini-api/docs/live#session-resumption). This feature helps prevent session termination when the server periodically resets the WebSocket connection.

### Configuration

Session resumption can be configured using the following parameter:

```json
{
  "enable_session_resumption": true
}
```

By default, session resumption is **enabled** (`true`).

### How it Works

1. **Session Setup**: When session resumption is enabled, the extension configures the session to receive resumption updates from the server.

2. **Handle Storage**: The server periodically sends `SessionResumptionUpdate` messages containing a resumption handle. The extension automatically stores the latest handle.

3. **Automatic Reconnection**: When reconnecting (due to network issues or server resets), the extension automatically uses the stored resumption handle to resume the previous session context.

4. **GoAway Handling**: The extension handles `GoAway` messages from the server, which signal that the connection will soon be terminated, allowing for graceful preparation.

### Benefits

- **Improved Reliability**: Sessions can survive temporary network interruptions and server-side connection resets
- **Context Preservation**: Previous conversation context is maintained across reconnections
- **Seamless User Experience**: Users don't notice when reconnections happen
- **Better Resource Utilization**: Avoids losing conversation history and context

### Logs

When session resumption is active, you'll see logs like:

```
Starting new session with resumption enabled
Session resumption handle updated: abcd1234efgh5678...
Server sent GoAway message. Connection will be terminated in 10000ms
Starting connection with session resumption handle: abcd1234efgh5678...
```

### Configuration Example

```json
{
  "api_key": "your_gemini_api_key",
  "model": "gemini-2.0-flash-live-001",
  "enable_session_resumption": true,
  "language": "en-US",
  "voice": "Puck",
  "server_vad": true,
  "transcribe_user": true,
  "transcribe_agent": true
}
```

## Other Configuration Options

- `enable_session_resumption`: Enable/disable session resumption (default: `true`)
- `audio_buffer_threshold`: Audio buffer size threshold for processing (default: `1024`)
- `start_of_speech_sensitivity`: VAD sensitivity for speech start detection
- `end_of_speech_sensitivity`: VAD sensitivity for speech end detection
- `prefix_padding_ms`: Padding before speech detection
- `silence_duration_ms`: Silence duration for speech end detection
- `affective_dialog`: Enable affective dialog features
- `proactive_audio`: Enable proactive audio responses

## Dependencies

This extension requires:
- `google-genai` Python package
- `websockets` for WebSocket communication
- `PIL` for image processing
- `numpy` for audio processing

## Note

Session resumption is automatically managed by the extension and requires no additional user intervention. The feature is designed to work transparently in the background to improve connection reliability.

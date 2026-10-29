# **Technical Blueprint: AI Technical Interviewer (Local MVP)**

This document outlines the architecture and step-by-step implementation guide for an AI employee capable of joining a Google Meet, conducting a technical interview (asking questions, listening, and responding dynamically), and providing a post-interview assessment. The system draws inspiration from the *Anthropic/AnthropicInterviewer* dataset methodology (which scales qualitative interviews using AI) but is adapted specifically for technical candidate evaluations running entirely locally using open-source tools.

## **1\. Technology Stack (Local MVP)**

> * **LLM Backend:** **Ollama** running a highly capable local model (e.g., llama3.1:8b or mistral) for fast inference.  
> * **Context & Tooling:** **Ollmcp (Model Context Protocol for Ollama)** to provide the AI with external context, such as the candidate's resume, the job description, and the question bank.  
> * **Containerization:** **Docker** & **Docker Compose** to orchestrate the services and manage complex audio dependencies cleanly.  
> * **Google Meet Integration:** **Playwright** (Python) running a headless Chromium browser inside a container, utilizing virtual audio devices to bridge audio streams.  
> * **Speech-to-Text (STT):** **Faster-Whisper** combined with **Silero VAD** (Voice Activity Detection) to transcribe candidate responses locally in real-time.  
> * **Text-to-Speech (TTS):** **Piper TTS** or **Coqui TTS** for low-latency, natural-sounding local voice generation.

## **2\. Architecture Design**

The system operates in a continuous loop during the interview, managed by a Python orchestrator. Here is the flow of data:

> 1. **Meet Joiner:** A headless Chromium instance accesses the Google Meet URL. It bypasses the lobby (once admitted by the host) and uses specific launch flags to route audio to and from virtual devices.  
> 2. **Audio Pipeline:**  
   * *Inbound (Listening):* The browser's audio output is captured via a PulseAudio virtual sink and streamed to the local Whisper service. VAD detects when the candidate finishes speaking.  
   * *Outbound (Speaking):* The generated TTS audio is routed to a PulseAudio virtual microphone, which the headless browser uses as its input device.  
> 3. **Conversational Agent:** The transcribed text is sent to the Python orchestrator, which maintains conversational history. It queries the local LLM via Ollama and streams the response text back to the TTS engine.  
> 4. **Post-Interview Assessment:** Once the session ends, the full transcript is fed back into Ollama with a specialized evaluation prompt to generate a final scorecard.

## **3\. Step-by-Step Implementation Guide**

### **Phase 1: Environment & Docker Setup**

Create a docker-compose.yml file to spin up the necessary components. You will need containers for Ollama, your Python orchestrator (which installs Playwright and audio dependencies), and the STT/TTS services if you modularize them.  
`version: '3.8'`  
`services:`  
  `ollama:`  
    `image: ollama/ollama`  
    `ports:`  
      `- "11434:11434"`  
    `volumes:`  
      `- ollama_data:/root/.ollama`  
    `deploy:`  
      `resources:`  
        `reservations:`  
          `devices:`  
            `- driver: nvidia`  
              `count: 1`  
              `capabilities: [gpu]`

  `interviewer-bot:`  
    `build: .`  
    `depends_on:`  
      `- ollama`  
    `environment:`  
      `- OLLAMA_HOST=http://ollama:11434`  
    `devices:`  
      `- /dev/snd:/dev/snd`  
    `volumes:`  
      `- ./:/app`  
`volumes:`  
  `ollama_data:`

### **Phase 2: Local AI & MCP Setup**

Configure Ollmcp to give the model access to the interview tools. Define an MCP server that exposes functions like get\_candidate\_resume() and get\_next\_technical\_question(). When the Python orchestrator initializes the Ollama client, bind it to this MCP server so the model can pull the required context dynamically during the interview.

### **Phase 3: The Headless Meet Bot (Playwright)**

To join the Google Meet without a physical display or microphone, configure Playwright with the following Chromium flags to inject a fake media stream. You will use Linux virtual audio loopbacks (e.g., v4l2loopback and PulseAudio null sinks) inside the Docker container.  
`import asyncio`  
`from playwright.async_api import async_playwright`

`async def join_meet(meet_url):`  
    `async with async_playwright() as p:`  
        `browser = await p.chromium.launch(`  
            `headless=True,`  
            `args=[`  
                `'--use-fake-ui-for-media-stream',`  
                `'--use-fake-device-for-media-stream',`  
                `'--use-file-for-fake-audio-capture=/app/outbound_audio.wav' # Stream TTS here`  
            `]`  
        `)`  
        `context = await browser.new_context(permissions=['camera', 'microphone'])`  
        `page = await context.new_page()`  
        `await page.goto(meet_url)`  
          
        `# Logic to click "Join now" and handle the lobby`  
        `await page.click('button:has-text("Ask to join")')`  
        `# Wait for host to admit...`

### **Phase 4: Audio Processing (STT & TTS)**

> * **STT (Listening):** Continuously record the audio output of the Google Meet tab. Run this audio stream through Silero VAD. When a pause of \>1.5 seconds is detected, send the audio chunk to faster-whisper for transcription.  
> * **TTS (Speaking):** When Ollama generates a response, stream the text to Piper TTS. Output the generated audio to the virtual microphone device that Playwright is capturing (or write to the file being used for fake audio capture).

### **Phase 5: Interview Orchestration Logic**

Inject a robust system prompt modeled after the *AnthropicInterviewer* methodology, but tailored for a Senior Engineering persona. The orchestrator must track the state of the interview:

> 1. **Introduction:** Greet the candidate and explain the format.  
> 2. **Resume Deep Dive:** Ask about a specific project (retrieved via MCP).  
> 3. **Technical Questions:** Deliver coding/system design questions and evaluate the verbal responses.  
> 4. **Sign-off:** Allow the candidate to ask questions, then conclude the call.

### **Phase 6: Post-Interview Assessment**

After the headless browser disconnects, aggregate the conversation log into a structured format (similar to the Anthropic dataset's work\_00XX text format). Prompt Ollama with the full transcript to generate a JSON assessment.  
`Prompt:`  
`"You are a hiring manager reviewing the following technical interview transcript.`   
`Provide a JSON assessment containing:`  
`1. 'technical_score': (1-10)`  
`2. 'communication_score': (1-10)`  
`3. 'strengths': [list]`  
`4. 'weaknesses': [list]`  
`5. 'hire_recommendation': 'Hire' or 'No Hire'`  
`6. 'summary': A brief paragraph."`

## **4\. Challenges & Mitigations**

| Challenge | Mitigation Strategy |
| :---- | :---- |
| **Latency (Awkward Silences)** | Stream the LLM response tokens directly to the TTS engine so it begins speaking before the LLM finishes generating the full sentence. Use quantized models (4-bit/8-bit). |
| **Interruptions (Barge-in)** | The orchestrator must monitor the VAD while the TTS is playing. If the candidate speaks loudly, immediately halt the TTS playback, log the interruption, and feed the new input back to Ollama. |
| **Hallucinations during evaluation** | Ground the model heavily using the MCP context. Force JSON-schema validation for the post-interview assessment to ensure the output is structured correctly. |


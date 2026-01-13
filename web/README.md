# Kiosk Agent UI

Modern web dashboard for the Kiosk Agent - Vision-Language-Action automation system.

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

Create `.env.local`:

```bash
# Backend API URL
NEXT_PUBLIC_BACKEND_URL=http://localhost:8080
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
web/
├── app/                      # Next.js App Router
│   ├── page.tsx              # Main page (redirects to demo)
│   ├── demo/
│   │   └── page.tsx          # Agent dashboard
│   ├── layout.tsx            # Root layout
│   └── globals.css           # Global styles
├── components/               # React components
│   ├── Header.tsx            # Navigation header
│   ├── ChatPanel.tsx         # Chat interface
│   ├── ThoughtPanel.tsx      # Agent reasoning display
│   ├── ActionLog.tsx         # ADB command log
│   ├── StatePanel.tsx        # Agent state display
│   ├── InterruptChoiceCard.tsx  # HITL choice card
│   ├── TTSPlayer.tsx         # Text-to-speech player
│   ├── GridScan.tsx          # Background animation
│   └── ui/                   # shadcn/ui components
├── lib/
│   └── utils.ts              # Utility functions
└── public/
    └── images/               # Static assets
```

## Features

- **Real-time Agent Monitoring**: Stream agent thoughts and actions
- **Human-in-the-Loop**: Interactive choice cards for agent questions
- **Voice Interface**: STT for input, TTS for agent responses
- **Dark Theme**: Modern, minimal aesthetic
- **Responsive Design**: Works on desktop and tablet

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion
- **UI Components**: shadcn/ui + Radix
- **Icons**: Lucide React

## API Integration

The dashboard connects to the backend via SSE (Server-Sent Events):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent/start` | POST | Start agent execution |
| `/agent/respond` | POST | Send HITL response |
| `/agent/interrupt` | POST | Force interrupt |
| `/stt/transcribe` | POST | Speech to text |
| `/tts/audio/:file` | GET | Get TTS audio |

## Development

```bash
# Development with hot reload
npm run dev

# Type checking
npm run lint

# Production build
npm run build

# Start production server
npm start
```

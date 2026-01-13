/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      // Agent SSE endpoints (direct, no CopilotKit)
      {
        source: '/api/agent/:path*',
        destination: 'http://localhost:8080/agent/:path*',
      },
      // STT endpoint
      {
        source: '/api/stt/:path*',
        destination: 'http://localhost:8080/stt/:path*',
      },
      // CopilotKit (kept for compatibility)
      {
        source: '/api/copilotkit/:path*',
        destination: 'http://localhost:8080/copilotkit/:path*',
      },
    ];
  },
};

module.exports = nextConfig;


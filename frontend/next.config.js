/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          // "same-origin-allow-popups" (not "same-origin") so the Google sign-in popup
          // can hand the result back to this window. A strict "same-origin" COOP severs
          // window.opener and Firebase's /__/auth/handler shows "The requested action is
          // invalid". No COEP header: nothing here needs cross-origin isolation, and
          // "require-corp" blocks Firebase's auth iframe.
          { key: "Cross-Origin-Opener-Policy", value: "same-origin-allow-popups" },
        ],
      },
    ];
  },
};

module.exports = nextConfig;

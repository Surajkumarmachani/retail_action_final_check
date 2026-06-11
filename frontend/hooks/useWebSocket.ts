// hooks/useWebSocket.ts
// Authenticated WebSocket connection + frame streaming loop.
//
// Protocol:
//   → { token }                   first message (auth handshake)
//   ← { type: "auth_ok",  uid }   server confirms auth
//   ← { type: "auth_error", ... } server rejects → connection closed
//   → { type: "frame", data: <b64 jpeg> }   every 1000/fps ms
//   ← { type: "predictions", data: [...] }  server reply per frame

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { auth } from "@/lib/firebase";
import { Prediction, ServerMessage } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL!;  // e.g. wss://api.example.com/ws/predict

export function usePredictorSocket(opts: {
  onPredictions: (preds: Prediction[]) => void;
  captureFrame:  () => string | null;
  webcamReady:   boolean;
  fps?:          number;
}) {
  const { onPredictions, captureFrame, webcamReady, fps = 15 } = opts;

  const wsRef       = useRef<WebSocket | null>(null);
  const timerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const [connected, setConnected]   = useState(false);
  const [authError, setAuthError]   = useState<string | null>(null);

  const stopStreaming = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const startStreaming = useCallback(() => {
    stopStreaming();
    timerRef.current = setInterval(() => {
      const frame = captureFrame();
      const ws    = wsRef.current;
      if (frame && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "frame", data: frame }));
      }
    }, Math.floor(1000 / fps));
  }, [captureFrame, fps, stopStreaming]);

  const connect = useCallback(async () => {
    const user = auth.currentUser;
    if (!user || wsRef.current?.readyState === WebSocket.OPEN) return;

    const token = await user.getIdToken(/* forceRefresh */ true);
    const ws    = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ token }));
    };

    ws.onmessage = (e: MessageEvent) => {
      const msg: ServerMessage = JSON.parse(e.data as string);
      if (msg.type === "auth_ok") {
        setConnected(true);
      } else if (msg.type === "auth_error") {
        setAuthError(msg.detail ?? "Authentication failed");
        ws.close();
      } else if (msg.type === "predictions" && msg.data) {
        onPredictions(msg.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      stopStreaming();
      // Auto-reconnect after 2 s (handles transient network drops)
      setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
  }, [onPredictions, stopStreaming]);

  // Start/stop frame streaming when connection state changes
  useEffect(() => {
    if (connected && webcamReady) {
      startStreaming();
    } else {
      stopStreaming();
    }
  }, [connected, webcamReady, startStreaming, stopStreaming]);

  // Connect whenever a Firebase user is available
  useEffect(() => {
    const unsub = auth.onAuthStateChanged((user) => {
      if (user) connect();
    });
    return () => {
      unsub();
      stopStreaming();
      wsRef.current?.close();
    };
  }, [connect, stopStreaming]);

  return { connected, authError };
}

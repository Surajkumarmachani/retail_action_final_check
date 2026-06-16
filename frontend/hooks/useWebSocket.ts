"use client";

// Fire-on-reply frame streaming:
//   Instead of setInterval (which queues frames even when backend is slow),
//   we fire the next frame immediately after receiving the server's response.
//   This keeps exactly 1 frame in-flight at all times — minimum possible latency.
//   A 500ms fallback interval restarts the chain if a response is ever missed.

import { useCallback, useEffect, useRef, useState } from "react";
import { auth } from "@/lib/firebase";
import { Prediction, ProductDetection, ServerMessage } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL!;

export function usePredictorSocket(opts: {
  onPredictions: (preds: Prediction[]) => void;
  onProducts?:   (prods: ProductDetection[]) => void;
  captureFrame:  () => string | null;
  webcamReady:   boolean;
  fps?:          number;
}) {
  const { onPredictions, onProducts, captureFrame, webcamReady } = opts;

  const wsRef       = useRef<WebSocket | null>(null);
  const timerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const inFlightRef = useRef(false);

  // Stable refs so ws.onmessage never captures stale closures
  const captureRef  = useRef(captureFrame);
  const onPredsRef  = useRef(onPredictions);
  const onProdsRef  = useRef(onProducts);
  useEffect(() => { captureRef.current  = captureFrame;  }, [captureFrame]);
  useEffect(() => { onPredsRef.current  = onPredictions; }, [onPredictions]);
  useEffect(() => { onProdsRef.current  = onProducts;    }, [onProducts]);

  const [connected, setConnected] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const sendFrame = useCallback(() => {
    if (inFlightRef.current) return;          // don't stack up frames
    const frame = captureRef.current();
    const ws    = wsRef.current;
    if (frame && ws?.readyState === WebSocket.OPEN) {
      inFlightRef.current = true;
      ws.send(JSON.stringify({ type: "frame", data: frame }));
    }
  }, []);

  const stopStreaming = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    inFlightRef.current = false;
  }, []);

  const startStreaming = useCallback(() => {
    stopStreaming();
    sendFrame();                             // kick off the fire-on-reply chain
    // Fallback: if the chain stalls (dropped response), restart after 500ms
    timerRef.current = setInterval(() => {
      inFlightRef.current = false;
      sendFrame();
    }, 500);
  }, [sendFrame, stopStreaming]);

  const connect = useCallback(async () => {
    const user = auth.currentUser;
    if (!user || wsRef.current?.readyState === WebSocket.OPEN) return;

    const token = await user.getIdToken(true);
    const ws    = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => { ws.send(JSON.stringify({ token })); };

    ws.onmessage = (e: MessageEvent) => {
      const msg: ServerMessage = JSON.parse(e.data as string);
      if (msg.type === "auth_ok") {
        setConnected(true);
      } else if (msg.type === "auth_error") {
        setAuthError(msg.detail ?? "Authentication failed");
        ws.close();
      } else if (msg.type === "predictions") {
        onPredsRef.current(msg.data ?? []);
        if (msg.products) onProdsRef.current?.(msg.products);
        // Response received — immediately fire next frame
        inFlightRef.current = false;
        sendFrame();
      }
    };

    ws.onclose = () => {
      setConnected(false);
      stopStreaming();
      setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
  }, [sendFrame, stopStreaming]);

  useEffect(() => {
    if (connected && webcamReady) startStreaming();
    else stopStreaming();
  }, [connected, webcamReady, startStreaming, stopStreaming]);

  useEffect(() => {
    const unsub = auth.onAuthStateChanged(user => { if (user) connect(); });
    return () => { unsub(); stopStreaming(); wsRef.current?.close(); };
  }, [connect, stopStreaming]);

  return { connected, authError };
}

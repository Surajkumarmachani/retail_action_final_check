"use client";

// Fire-on-reply frame streaming:
//   Instead of setInterval (which queues frames even when backend is slow),
//   we fire the next frame immediately after receiving the server's response.
//   This keeps exactly 1 frame in-flight at all times — minimum possible latency.
//   A 500ms fallback interval restarts the chain if a response is ever missed.

import { useCallback, useEffect, useRef, useState } from "react";
import { auth } from "@/lib/firebase";
import { Prediction, ProductDetection, BAMEvent, ServerMessage } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL!;

export function usePredictorSocket(opts: {
  onPredictions: (preds: Prediction[]) => void;
  onProducts?:   (prods: ProductDetection[]) => void;
  onEvents?:     (evs: BAMEvent[]) => void;
  captureFrame:  () => string | null;
  webcamReady:   boolean;
  fps?:          number;
}) {
  const { onPredictions, onProducts, onEvents, captureFrame, webcamReady } = opts;

  const wsRef       = useRef<WebSocket | null>(null);
  const timerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const inFlightRef = useRef(false);
  const lastSentRef = useRef(0);   // when the in-flight frame was sent (for the stall watchdog)

  // Stable refs so ws.onmessage never captures stale closures
  const captureRef  = useRef(captureFrame);
  const onPredsRef  = useRef(onPredictions);
  const onProdsRef  = useRef(onProducts);
  const onEventsRef = useRef(onEvents);
  useEffect(() => { captureRef.current  = captureFrame;  }, [captureFrame]);
  useEffect(() => { onPredsRef.current  = onPredictions; }, [onPredictions]);
  useEffect(() => { onProdsRef.current  = onProducts;    }, [onProducts]);
  useEffect(() => { onEventsRef.current = onEvents;      }, [onEvents]);

  const [connected, setConnected] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const sendFrame = useCallback(() => {
    if (inFlightRef.current) return;          // STRICTLY 1 frame in flight — never buffer
    const frame = captureRef.current();
    const ws    = wsRef.current;
    if (frame && ws?.readyState === WebSocket.OPEN) {
      inFlightRef.current = true;
      lastSentRef.current = Date.now();
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
    // Stall watchdog: ONLY resend if a frame has been in flight with no reply for >1.5s
    // (a genuinely dropped response). Never force-sends otherwise — this keeps exactly
    // one frame in flight so the backend always processes the FRESHEST frame (no lag buildup).
    timerRef.current = setInterval(() => {
      if (inFlightRef.current && Date.now() - lastSentRef.current > 1500) {
        inFlightRef.current = false;
        sendFrame();
      } else if (!inFlightRef.current) {
        sendFrame();                         // restart chain if it ever idled out
      }
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
        if (msg.events && msg.events.length) onEventsRef.current?.(msg.events);
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

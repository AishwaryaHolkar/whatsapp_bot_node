// =============================================================================
// WhatsApp AI Chatbot — single entry point
// Flow: User sends WhatsApp message → Meta calls POST /webhook → we reply 200
//       immediately → OpenAI generates an answer → we send it back via Meta API
// =============================================================================

import "dotenv/config";
import express from "express";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import QRCode from "qrcode";
import { waitUntil } from "@vercel/functions"; // keeps Vercel function alive after we reply 200

// ----------------------------------------------------------------------------
// 1. CONFIG — read everything from .env and stop early if something is missing
// ----------------------------------------------------------------------------
const {
  PORT = 3000,
  ACCESS_TOKEN,          // Meta token used to SEND messages
  APP_SECRET,            // Meta app secret used to VERIFY incoming webhooks
  PHONE_NUMBER_ID,       // ID of the business number that sends replies
  TEST_PHONE_NUMBER,     // Business phone number (digits only) shown in the QR code
  VERIFY_TOKEN,          // Any secret string; must match the one in Meta dashboard
  OPENAI_API_KEY,
  OPENAI_MODEL = "gpt-4o-mini",
  GRAPH_API_VERSION = "v21.0",
  SYSTEM_PROMPT = "You are a helpful and polite customer support assistant. Keep replies short and clear.",
} = process.env;

const required = { ACCESS_TOKEN, APP_SECRET, PHONE_NUMBER_ID, TEST_PHONE_NUMBER, VERIFY_TOKEN, OPENAI_API_KEY };
const missing = Object.entries(required).filter(([, v]) => !v).map(([k]) => k);
if (missing.length) {
  console.error(`❌ Missing in .env: ${missing.join(", ")}`);
  process.exit(1);
}

// Keep only digits: "+91 98765-43210" → "919876543210" (wa.me needs digits only)
const BUSINESS_NUMBER = TEST_PHONE_NUMBER.replace(/\D/g, "");
const WHATSAPP_LINK = `https://wa.me/${BUSINESS_NUMBER}`;

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ----------------------------------------------------------------------------
// 2. IN-MEMORY STORES (fine for local testing; use Redis in production)
// ----------------------------------------------------------------------------
const processedMessageIds = new Set(); // stops duplicate replies when Meta retries
const chatHistory = new Map();          // phone number → last few messages
const MAX_HISTORY = 10;                 // keep last 10 messages per user

// ----------------------------------------------------------------------------
// 3. EXPRESS APP
// ----------------------------------------------------------------------------
const app = express();

// Save the RAW body. Signature check must use the exact bytes Meta sent,
// not the parsed JSON, otherwise the check always fails.
app.use(express.json({ verify: (req, _res, buf) => { req.rawBody = buf; } }));

// ----------------------------------------------------------------------------
// GET /  → info page (index.html) with the business number and QR code filled in
// ----------------------------------------------------------------------------
app.get("/", async (_req, res) => {
  try {
    const qrDataUrl = await QRCode.toDataURL(WHATSAPP_LINK, { width: 260, margin: 1 });
    const html = fs
      .readFileSync(path.join(__dirname, "public", "index.html"), "utf8")
      .replaceAll("{{QR_CODE}}", qrDataUrl)
      .replaceAll("{{WHATSAPP_LINK}}", WHATSAPP_LINK)
      .replaceAll("{{BUSINESS_NUMBER}}", `+${BUSINESS_NUMBER}`)
      .replaceAll("{{MODEL}}", OPENAI_MODEL);
    res.type("html").send(html);
  } catch (err) {
    console.error("Page render error:", err);
    res.status(500).send("Could not load page");
  }
});

// ----------------------------------------------------------------------------
// GET /health → simple check that the server is running
// ----------------------------------------------------------------------------
app.get("/health", (_req, res) => res.json({ status: "ok", model: OPENAI_MODEL }));

// ----------------------------------------------------------------------------
// GET /webhook → ONE-TIME verification when you add the webhook URL in Meta.
// Meta sends hub.challenge; we return it only if the verify token matches.
// ----------------------------------------------------------------------------
app.get("/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    console.log("✅ Webhook verified by Meta");
    return res.status(200).send(challenge);
  }
  console.warn("⚠️ Webhook verification failed (token mismatch)");
  res.sendStatus(403);
});

// ----------------------------------------------------------------------------
// POST /webhook → Meta calls this for EVERY event (new message, delivered, read)
// ----------------------------------------------------------------------------
app.post("/webhook", (req, res) => {
  // Step 1: make sure the request really came from Meta
  if (!isValidSignature(req)) {
    console.warn("⚠️ Invalid signature — request rejected");
    return res.sendStatus(401);
  }

  // Step 2: answer Meta immediately. If we wait for OpenAI, Meta may retry
  // and the user gets duplicate replies.
  res.sendStatus(200);

  // Step 3: do the real work in the background.
  // waitUntil tells Vercel: "don't freeze this function until this promise finishes".
  // Locally (npm start) it has no effect and the promise simply runs as normal.
  waitUntil(handleWebhookEvent(req.body).catch((err) => console.error("Handler error:", err)));
});

// ----------------------------------------------------------------------------
// Checks the X-Hub-Signature-256 header using APP_SECRET (HMAC SHA-256)
// ----------------------------------------------------------------------------
function isValidSignature(req) {
  const header = req.headers["x-hub-signature-256"];
  if (!header || !req.rawBody) return false;

  const expected = "sha256=" + crypto.createHmac("sha256", APP_SECRET).update(req.rawBody).digest("hex");
  const a = Buffer.from(header);
  const b = Buffer.from(expected);
  // timingSafeEqual prevents timing attacks; lengths must match first
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

// ----------------------------------------------------------------------------
// Reads the webhook body, finds the user message, and sends a reply
// ----------------------------------------------------------------------------
async function handleWebhookEvent(body) {
  const value = body?.entry?.[0]?.changes?.[0]?.value;
  const message = value?.messages?.[0];

  // No message = status update (sent / delivered / read). Ignore it.
  if (!message) return;

  // Same message ID seen before = Meta retry. Ignore it.
  if (processedMessageIds.has(message.id)) return;
  processedMessageIds.add(message.id);
  if (processedMessageIds.size > 5000) processedMessageIds.clear(); // avoid memory growth

  const userNumber = message.from; // user's private number (digits)

  // Only text is supported. Anything else gets a polite fixed reply.
  if (message.type !== "text") {
    await sendWhatsAppText(
      userNumber,
      "Thank you for your message! 🙏 Right now I can only read text messages. Please type your question and I'll be happy to help."
    );
    return;
  }

  const userText = message.text.body;
  console.log(`📩 ${userNumber}: ${userText}`);

  try {
    const reply = await getAIReply(userNumber, userText);
    await sendWhatsAppText(userNumber, reply);
    console.log(`📤 Reply sent to ${userNumber}`);
  } catch (err) {
    console.error("AI/send error:", err.message);
    await sendWhatsAppText(
      userNumber,
      "Sorry, I'm having a small problem right now. Please try again in a moment. 🙏"
    ).catch(() => {});
  }
}

// ----------------------------------------------------------------------------
// Sends the user message + recent history to OpenAI and returns the answer
// ----------------------------------------------------------------------------
async function getAIReply(userNumber, userText) {
  const history = chatHistory.get(userNumber) || [];
  history.push({ role: "user", content: userText });

  const completion = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    messages: [{ role: "system", content: SYSTEM_PROMPT }, ...history],
  });

  const reply = completion.choices[0]?.message?.content?.trim() || "Sorry, I could not understand. Could you rephrase?";

  history.push({ role: "assistant", content: reply });
  chatHistory.set(userNumber, history.slice(-MAX_HISTORY)); // keep only recent messages
  return reply;
}

// ----------------------------------------------------------------------------
// Sends a text message to a WhatsApp user through Meta Graph API
// ----------------------------------------------------------------------------
async function sendWhatsAppText(to, text) {
  const url = `https://graph.facebook.com/${GRAPH_API_VERSION}/${PHONE_NUMBER_ID}/messages`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ACCESS_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messaging_product: "whatsapp",
      to,
      type: "text",
      text: { body: text.slice(0, 4096) }, // WhatsApp text limit is 4096 characters
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Meta API ${response.status}: ${error}`);
  }
}

// ----------------------------------------------------------------------------
// 4. START SERVER
// On Vercel, Vercel runs the app itself (process.env.VERCEL is set) — no listen().
// Locally, we start a normal server on PORT.
// ----------------------------------------------------------------------------
if (!process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`🤖 Model: ${OPENAI_MODEL}`);
    console.log(`📱 WhatsApp link: ${WHATSAPP_LINK}`);
  });
}

export default app; // Vercel uses this exported app as the request handler

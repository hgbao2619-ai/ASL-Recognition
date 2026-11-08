import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Khởi tạo Gemini AI Client với Google GenAI SDK
function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey.length < 5) {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Endpoint status để kiểm tra trạng thái Gemini Flash Lite
app.get('/api/gemini/status', (req, res) => {
  const client = getGeminiClient();
  res.json({
    model: 'gemini-3.1-flash-lite',
    isAvailable: Boolean(client),
    hasKey: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'),
  });
});

// Endpoint: Gemini Flash Lite chuyển đổi các chữ cái rời rạc thành câu hoàn chỉnh
app.post('/api/gemini/sentence', async (req, res) => {
  try {
    const { characters, language = 'en' } = req.body;
    if (!characters || !Array.isArray(characters) || characters.length === 0) {
      return res.status(400).json({ error: 'No characters provided' });
    }

    const rawChars = characters.join('').trim();
    if (!rawChars) {
      return res.json({ sentence: '', source: 'empty' });
    }

    const upper = rawChars.toUpperCase();
    const upperNoSpaces = rawChars.replace(/\s+/g, '').toUpperCase();
    const isVi = language === 'vi';
    const targetLangName = isVi ? 'Vietnamese (Tiếng Việt)' : 'English';

    // 1. GỌI GEMINI FLASH LITE LLM (gemini-3.1-flash-lite)
    const aiClient = getGeminiClient();
    if (aiClient) {
      try {
        const systemInstruction = `You are an expert American Sign Language (ASL) and fingerspelling translation assistant powered by Gemini Flash Lite.
Your task is to take discrete fingerspelled letters or segmented words from ASL hand sign detection, correct any minor typo or missing character, and synthesize them into a single fluent, natural, grammatically correct, polite ${targetLangName} sentence for daily communication.

CRITICAL RULES:
1. Output ONLY the final generated sentence in ${targetLangName}.
2. Do NOT wrap the result in quotes or add conversational preamble.
3. If the input is an acronym or greeting (e.g., "ILY", "HI", "DAY", "FOOD", "HOLD", "FLY"), expand it into a warm, natural sentence.`;

        const userPrompt = `Fingerspelled input: "${rawChars}"
Target language: ${targetLangName}

Reconstruct these discrete letters into a natural and complete sentence in ${targetLangName}.`;

        const geminiPromise = aiClient.models.generateContent({
          model: 'gemini-3.1-flash-lite',
          contents: userPrompt,
          config: {
            systemInstruction,
          },
        });

        // 12s timeout for reliable response
        const timeoutPromise = new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Gemini Flash Lite timeout')), 12000)
        );

        const response: any = await Promise.race([geminiPromise, timeoutPromise]);
        let sentence = response.text?.trim() || '';
        sentence = sentence.replace(/^["']|["']$/g, '').trim();

        if (sentence) {
          return res.json({
            sentence,
            source: 'gemini-3.1-flash-lite',
            model: 'gemini-3.1-flash-lite',
          });
        }
      } catch (geminiError: any) {
        console.warn('Gemini Flash Lite API call notice:', geminiError?.message || geminiError);
      }
    }

    // 2. Fallback Dictionary & Heuristics (khi chưa cấu hình API key hoặc offline)
    const dictionaryEn: Record<string, string> = {
      'HI': 'Hi there! How are you doing today?',
      'ILY': 'I love you very much.',
      'I L Y': 'I love you very much.',
      'DAY': 'Have a wonderful day!',
      'FLY': 'The bird can fly high in the sky.',
      'FOOD': 'Would you like to get some food together?',
      'HOLD': 'Please hold on for just a moment.',
      'CALL': 'Please give me a call when you are free.',
      'BOY': 'That boy is over there.',
      'CAR': 'I see a car coming.',
      'CHILL': 'Let us chill and relax together.',
      'OH': 'Oh, I see what you mean!',
      'OK': 'Everything is completely okay.',
      'BYE': 'Goodbye, hope to see you again soon!',
      'YOU': 'It is so wonderful to see you!',
      'BUY': 'I would like to buy this, please.',
      'CLUB': 'Welcome to the club!',
      'GOOD': 'Have a really good time!',
      'GO': 'Let us go together!',
      'FOX': 'Look at that clever fox.',
      'BOX': 'Open the mystery box.',
      'WOW': 'Wow, that is really amazing!',
      'WIN': 'We did it, we win!',
      'WE': 'We are all here together.',
      'CAT': 'The playful cat is purring softly.',
      'KID': 'The kid is smiling happily.',
      'THANK': 'Thank you so much for your support!',
      'NO': 'No, thank you very much.',
      'TIME': 'Now is the perfect time to begin.',
      'MOM': 'Mom, I love you so much!',
      'RUN': 'Let us run together under the morning sun.',
      'ROOM': 'Welcome to my cozy room.',
      'RAIN': 'The gentle rain is falling outside.',
      'MAN': 'That kind man offered his help.',
    };

    const dictionaryVi: Record<string, string> = {
      'HI': 'Xin chào! Bạn có khỏe không?',
      'ILY': 'Tôi yêu bạn rất nhiều.',
      'I L Y': 'Tôi yêu bạn rất nhiều.',
      'DAY': 'Chúc bạn một ngày thật tuyệt vời và tràn ngập niềm vui!',
      'FLY': 'Cánh chim đang bay lượn thật cao trên bầu trời.',
      'FOOD': 'Bạn có muốn cùng đi ăn chút gì không?',
      'HOLD': 'Xin vui lòng chờ một chút nhé.',
      'CALL': 'Hãy gọi điện cho tôi khi bạn rảnh rỗi nhé.',
      'BOY': 'Cậu bé kia đang đứng đằng ấy.',
      'CAR': 'Tôi thấy một chiếc xe hơi đang đi tới.',
      'CHILL': 'Hãy cùng thư giãn và nghỉ ngơi nhé.',
      'GOOD': 'Chúc bạn có một khoảng thời gian thật tuyệt vời!',
      'GO': 'Chúng ta cùng xuất phát thôi nào!',
      'FOX': 'Nhìn chú cáo thông minh kìa.',
      'BOX': 'Hãy mở chiếc hộp bất ngờ này ra nhé.',
      'OH': 'Ồ, tôi hiểu ý bạn rồi!',
      'OK': 'Mọi chuyện đều ổn cả.',
      'BYE': 'Tạm biệt, hẹn sớm gặp lại bạn nhé!',
      'YOU': 'Rất vui được gặp lại bạn!',
      'BUY': 'Tôi muốn mua món đồ này.',
      'CLUB': 'Chào mừng bạn đã đến với câu lạc bộ!',
      'WOW': 'Oa, điều đó thật là tuyệt vời!',
      'WIN': 'Chúng ta đã chiến thắng rồi!',
      'WE': 'Chúng ta đang ở bên nhau.',
      'CAT': 'Chú mèo đáng yêu đang cuộn tròn nằm ngủ.',
      'KID': 'Đứa trẻ đang nở nụ cười rạng rỡ.',
      'THANK': 'Cảm ơn bạn rất nhiều vì đã luôn đồng hành!',
      'NO': 'Không sao đâu, cảm ơn bạn rất nhiều.',
      'TIME': 'Bây giờ là thời điểm lý tưởng để bắt đầu.',
      'MOM': 'Mẹ ơi, con yêu mẹ rất nhiều!',
      'RUN': 'Chúng ta cùng chạy bộ dưới ánh nắng sớm nhé.',
      'ROOM': 'Chào mừng bạn đến với căn phòng ấm cúng này.',
      'RAIN': 'Cơn mưa rào dịu mát đang rơi ngoài hiên.',
      'MAN': 'Người đàn ông tốt bụng đó đã nhiệt tình giúp đỡ chúng tôi.',
    };

    const dict = isVi ? dictionaryVi : dictionaryEn;

    if (dict[upper]) {
      return res.json({ sentence: dict[upper], source: 'fallback_dictionary', model: 'rule_based' });
    }
    if (dict[upperNoSpaces]) {
      return res.json({ sentence: dict[upperNoSpaces], source: 'fallback_dictionary', model: 'rule_based' });
    }

    // Heuristic sentence formatter for any other arbitrary letter sequence
    const words = rawChars.split(/\s+/).filter(Boolean);
    const formattedSentence = words
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
      .join(' ') + '.';

    return res.json({
      sentence: formattedSentence,
      source: 'heuristic_builder',
      model: 'local_engine',
    });

  } catch (error: any) {
    console.error('Sentence generation error:', error);
    return res.status(500).json({ error: error?.message || 'Server error' });
  }
});

// Development vs Production serving
async function setupServer() {
  const isProduction = process.env.NODE_ENV === 'production';

  if (!isProduction) {
    // Dynamic import vite in dev
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`[ASL SignLens Server] Running on http://localhost:${PORT} in ${isProduction ? 'production' : 'development'} mode`);
  });
}

setupServer();

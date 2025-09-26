const express = require('express');
const cors = require('cors');
const { google } = require('googleapis');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Set up storage for audio files
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const fileExt = path.extname(file.originalname);
    cb(null, `${uuidv4()}${fileExt}`);
  }
});

const upload = multer({ storage });

// Google OAuth2 setup
const oauth2Client = new google.auth.OAuth2(
  process.env.CLIENT_ID,
  process.env.CLIENT_SECRET,
  process.env.REDIRECT_URI
);

// For demo purposes, we'll use service account or pre-authenticated client
// In production, implement proper OAuth2 flow
const calendar = google.calendar({ version: 'v3', auth: oauth2Client });

// API endpoint to create meeting
app.post('/api/create-meeting', async (req, res) => {
  try {
    const { date, time, therapistId, note } = req.body;
    
    // Demo emails (in production, get from authenticated user)
    const userEmail = process.env.USER_EMAIL;
    const therapistEmail = process.env.THERAPIST_EMAIL;
    
    // Format date and time for Google Calendar
    const startDateTime = new Date(`${date}T${time}`);
    const endDateTime = new Date(startDateTime.getTime() + 60 * 60 * 1000); // 1 hour session
    
    // Create calendar event with Google Meet
    const event = {
      summary: 'Therapy Session',
      description: note || 'Therapy session',
      start: {
        dateTime: startDateTime.toISOString(),
        timeZone: 'Asia/Kolkata',
      },
      end: {
        dateTime: endDateTime.toISOString(),
        timeZone: 'Asia/Kolkata',
      },
      attendees: [
        { email: userEmail },
        { email: therapistEmail }
      ],
      conferenceData: {
        createRequest: {
          requestId: `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
          conferenceSolutionKey: { type: 'hangoutsMeet' }
        }
      }
    };
    
    const response = await calendar.events.insert({
      calendarId: 'primary',
      resource: event,
      conferenceDataVersion: 1,
      sendUpdates: 'all'
    });
    
    // Extract the Google Meet link
    const meetLink = response.data.hangoutLink || '';
    
    res.json({ 
      success: true, 
      meetLink,
      eventId: response.data.id
    });
    
  } catch (error) {
    console.error('Error creating meeting:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to create meeting',
      details: error.message
    });
  }
});

// API endpoint to process audio and get transcription
app.post('/api/process-audio', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No audio file provided' });
    }

    const audioFilePath = req.file.path;
    const language = req.body.language || 'en-US';
    const outputFileName = `transcript_${path.basename(audioFilePath, path.extname(audioFilePath))}.txt`;
    const outputFilePath = path.join(__dirname, 'transcripts', outputFileName);

    // Create transcripts directory if it doesn't exist
    const transcriptsDir = path.join(__dirname, 'transcripts');
    if (!fs.existsSync(transcriptsDir)) {
      fs.mkdirSync(transcriptsDir, { recursive: true });
    }

    // Run the Python script to process the audio
    const pythonProcess = spawn('python', [
      path.join(__dirname, '..', 'speechToText', 'speech_to_text.py'),
      audioFilePath,
      language,
      outputFilePath
    ]);

    let transcript = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Python output:', output);
      if (output.includes('Transcription:')) {
        transcript = output.split('Transcription:')[1].trim().split('\n')[0];
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
      console.error('Python error:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return res.status(500).json({ 
          success: false, 
          error: `Process exited with code ${code}: ${error}` 
        });
      }

      // If transcript wasn't captured from stdout, try to read from the file
      if (!transcript && fs.existsSync(outputFilePath)) {
        try {
          transcript = fs.readFileSync(outputFilePath, 'utf8');
        } catch (readError) {
          console.error('Error reading transcript file:', readError);
        }
      }

      // Simple sentiment analysis (placeholder for ML model)
      const sentimentScore = analyzeSentiment(transcript);

      res.json({
        success: true,
        transcript,
        outputFile: outputFilePath,
        analysis: {
          sentiment: sentimentScore,
          keywords: extractKeywords(transcript)
        }
      });
    });
  } catch (error) {
    console.error('Error processing audio:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to process audio'
    });
  }
});

// Simple sentiment analysis function (placeholder for ML model)
function analyzeSentiment(text) {
  const positiveWords = ['happy', 'good', 'great', 'excellent', 'joy', 'love', 'positive', 'wonderful', 'fantastic'];
  const negativeWords = ['sad', 'bad', 'terrible', 'awful', 'hate', 'negative', 'depressed', 'anxious', 'worried'];
  
  const words = text.toLowerCase().split(/\W+/);
  
  let positiveCount = 0;
  let negativeCount = 0;
  
  words.forEach(word => {
    if (positiveWords.includes(word)) positiveCount++;
    if (negativeWords.includes(word)) negativeCount++;
  });
  
  // Calculate a score between -1 and 1
  const totalWords = words.length;
  const score = totalWords > 0 ? 
    ((positiveCount - negativeCount) / Math.max(1, positiveCount + negativeCount)) : 0;
  
  return {
    score,
    label: score > 0.2 ? 'positive' : score < -0.2 ? 'negative' : 'neutral'
  };
}

// Extract keywords function (placeholder for ML model)
function extractKeywords(text) {
  const stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
                    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
                    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'];
  
  const words = text.toLowerCase().split(/\W+/).filter(word => 
    word.length > 3 && !stopWords.includes(word)
  );
  
  // Count word frequencies
  const wordFreq = {};
  words.forEach(word => {
    wordFreq[word] = (wordFreq[word] || 0) + 1;
  });
  
  // Sort by frequency
  const sortedWords = Object.keys(wordFreq).sort((a, b) => wordFreq[b] - wordFreq[a]);
  
  // Return top 5 keywords
  return sortedWords.slice(0, 5);
}

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
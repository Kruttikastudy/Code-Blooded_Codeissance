import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  Mic, 
  MicOff, 
  Play, 
  Pause, 
  Square,
  Volume2,
  Brain,
  Loader2,
  CheckCircle,
  ArrowRight
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const questions = [
  "How are you feeling today? Please describe your current mood and energy level.",
  "Can you tell me about something that brought you joy this week?",
  "Describe any challenges or stressors you've been experiencing lately.",
  "How has your sleep been over the past few days?",
  "What are you looking forward to in the coming week?",
];

export const VoiceQuiz = () => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState<string | null>(null);
  const [answers, setAnswers] = useState<string[]>([]);
  const [quizComplete, setQuizComplete] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const { toast } = useToast();

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setRecordedAudio(audioUrl);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      
      // Simulate audio level animation
      const interval = setInterval(() => {
        setAudioLevel(Math.random() * 100);
      }, 100);

      setTimeout(() => clearInterval(interval), 10000);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Recording Error",
        description: "Unable to access microphone. Please check permissions.",
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setAudioLevel(0);
    }
  };

  const processAnswer = () => {
    setIsProcessing(true);
    
    // Simulate AI processing
    setTimeout(() => {
      const newAnswers = [...answers, `Answer ${currentQuestion + 1} processed`];
      setAnswers(newAnswers);
      setRecordedAudio(null);
      setIsProcessing(false);
      
      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1);
      } else {
        setQuizComplete(true);
      }
    }, 3000);
  };

  const progress = ((currentQuestion + 1) / questions.length) * 100;

  if (quizComplete) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <Card className="text-center">
          <CardHeader>
            <div className="flex justify-center mb-4">
              <div className="p-4 bg-success/10 rounded-full">
                <CheckCircle className="h-12 w-12 text-success" />
              </div>
            </div>
            <CardTitle className="text-2xl">Analysis Complete!</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-6">
              Your voice analysis has been processed. The AI is now generating 
              your personalized mental health insights.
            </p>
            
            <div className="space-y-4 mb-8">
              <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg">
                <span className="font-medium">Speech Analysis</span>
                <CheckCircle className="h-5 w-5 text-success" />
              </div>
              <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg">
                <span className="font-medium">Sentiment Processing</span>
                <CheckCircle className="h-5 w-5 text-success" />
              </div>
              <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg">
                <span className="font-medium">Risk Assessment</span>
                <CheckCircle className="h-5 w-5 text-success" />
              </div>
            </div>

            <Button size="lg" className="px-8">
              View Your Results
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">Voice Analysis Quiz</h1>
          <Badge variant="outline">
            Question {currentQuestion + 1} of {questions.length}
          </Badge>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Question Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Volume2 className="h-5 w-5 text-primary" />
              <span>Question {currentQuestion + 1}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-primary/5 p-6 rounded-lg mb-6">
              <p className="text-lg leading-relaxed">
                {questions[currentQuestion]}
              </p>
            </div>
            
            <div className="text-sm text-muted-foreground space-y-2">
              <p>• Speak naturally and take your time</p>
              <p>• Aim for 30-60 seconds per response</p>
              <p>• Your voice patterns help us understand your mental state</p>
            </div>
          </CardContent>
        </Card>

        {/* Recording Interface */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Mic className="h-5 w-5 text-primary" />
              <span>Voice Recording</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            {!isRecording && !recordedAudio && !isProcessing && (
              <div className="space-y-6">
                <div className="p-8">
                  <div className="w-24 h-24 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
                    <Mic className="h-12 w-12 text-primary" />
                  </div>
                  <p className="text-muted-foreground mb-4">
                    Click to start recording your response
                  </p>
                </div>
                <Button 
                  size="lg" 
                  onClick={startRecording}
                  className="px-8"
                >
                  <Mic className="h-4 w-4 mr-2" />
                  Start Recording
                </Button>
              </div>
            )}

            {isRecording && (
              <div className="space-y-6">
                <div className="p-8">
                  <div className="w-24 h-24 mx-auto bg-destructive/10 rounded-full flex items-center justify-center mb-4 animate-pulse">
                    <MicOff className="h-12 w-12 text-destructive" />
                  </div>
                  <p className="text-lg font-semibold text-destructive mb-2">Recording...</p>
                  <p className="text-sm text-muted-foreground">
                    Speak clearly into your microphone
                  </p>
                </div>
                
                {/* Audio Level Visualization */}
                <div className="flex justify-center space-x-1 mb-6">
                  {Array.from({ length: 10 }).map((_, i) => (
                    <div
                      key={i}
                      className="w-2 bg-primary rounded-full transition-all duration-150"
                      style={{
                        height: `${Math.max(8, (audioLevel + i * 10) % 40)}px`,
                      }}
                    />
                  ))}
                </div>

                <Button 
                  size="lg" 
                  variant="destructive"
                  onClick={stopRecording}
                  className="px-8"
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop Recording
                </Button>
              </div>
            )}

            {recordedAudio && !isProcessing && (
              <div className="space-y-6">
                <div className="p-8">
                  <div className="w-24 h-24 mx-auto bg-success/10 rounded-full flex items-center justify-center mb-4">
                    <CheckCircle className="h-12 w-12 text-success" />
                  </div>
                  <p className="text-lg font-semibold text-success mb-2">Recording Complete</p>
                  <p className="text-sm text-muted-foreground">
                    Review your audio and submit for analysis
                  </p>
                </div>
                
                <audio controls className="w-full mb-4">
                  <source src={recordedAudio} type="audio/wav" />
                  Your browser does not support the audio element.
                </audio>

                <div className="flex space-x-3">
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      setRecordedAudio(null);
                      setCurrentQuestion(currentQuestion);
                    }}
                  >
                    Re-record
                  </Button>
                  <Button 
                    size="lg" 
                    onClick={processAnswer}
                    className="flex-1"
                  >
                    <Brain className="h-4 w-4 mr-2" />
                    Analyze Response
                  </Button>
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="space-y-6 p-8">
                <div className="w-24 h-24 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
                  <Loader2 className="h-12 w-12 text-primary animate-spin" />
                </div>
                <div>
                  <p className="text-lg font-semibold mb-2">Processing Audio...</p>
                  <p className="text-sm text-muted-foreground mb-4">
                    AI is analyzing your speech patterns, tone, and content
                  </p>
                </div>
                
                <div className="space-y-3 text-left">
                  <div className="flex items-center space-x-3">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span className="text-sm">Speech-to-text transcription</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span className="text-sm">Sentiment analysis</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span className="text-sm">Voice pattern recognition</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Progress Indicators */}
      <div className="mt-8">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Progress: {answers.length} of {questions.length} completed
              </div>
              <div className="flex space-x-2">
                {questions.map((_, index) => (
                  <div
                    key={index}
                    className={`w-3 h-3 rounded-full ${
                      index < answers.length
                        ? 'bg-success'
                        : index === currentQuestion
                        ? 'bg-primary'
                        : 'bg-muted'
                    }`}
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
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
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [answers, setAnswers] = useState<string[]>([]);
  const [quizComplete, setQuizComplete] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [processingQueue, setProcessingQueue] = useState<{ blob: Blob; questionIndex: number }[]>([]);
  const [scores, setScores] = useState<number[]>([]);
  const [allRecorded, setAllRecorded] = useState(false);
  const [showResultsModal, setShowResultsModal] = useState(false);
  const [showTranscript, setShowTranscript] = useState(false);
  const [showSOS, setShowSOS] = useState(false);


  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const { toast } = useToast();

  const startRecording = async () => {
    if (currentQuestion >= questions.length || answers[currentQuestion]) {
    toast({
      variant: "destructive",
      title: "Quiz Complete",
      description: "You have already answered all questions.",
    });
    return;
  }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setRecordedAudio(audioUrl);
        setRecordedBlob(audioBlob);
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
  if (!recordedBlob) {
    toast({
      variant: "destructive",
      title: "No Audio",
      description: "Please record your response before analyzing.",
    });
    return;
  }

  // Queue blob for background processing
  setProcessingQueue((prev) => [
    ...prev,
    { blob: recordedBlob, questionIndex: currentQuestion }
  ]);

  // Reset state for UI
  setRecordedAudio(null);
  setRecordedBlob(null);

  // Move to next question immediately
  if (currentQuestion < questions.length - 1) {
    setCurrentQuestion((prev) => prev + 1);
  } else {
    setQuizComplete(true);
    window.dispatchEvent(new CustomEvent("first-quiz-complete"));
  }
};




  const progress = ((currentQuestion + 1) / questions.length) * 100;

  useEffect(() => {
    if (allRecorded && answers.length==questions.length) {
      setQuizComplete(true);
    }
  }, [allRecorded, answers]);


  useEffect(() => {
  if (processingQueue.length === 0) return;

  const next = processingQueue[0];

  const processBlob = async (blob: Blob, questionIndex: number) => {
    const form = new FormData();
    form.append("audio", blob, "response.webm");
    form.append("language", "en-US");

    try {
      const res = await fetch("http://localhost:5000/api/process-audio", {
        method: "POST",
        body: form,
      });
      const data = await res.json();

      if (data.success && data.transcript) {
        setTranscript(data.transcript);
        setAnswers((prev) => {
          const newArr = [...prev];
          newArr[next.questionIndex] = data.transcript;
          return newArr;
        });
      }
      if (data.success && data.prediction) {
        const score = data.prediction.risk_score ?? 0; 
        setScores((prev) => {
          const newScores = [...prev];
          newScores[next.questionIndex] = score;
          return newScores;
        });
      }
    } catch (error) {
      console.error("Processing failed:", error);
    } finally {
      // Remove from queue
      setProcessingQueue((prev) => prev.slice(1));
    }
  };

  processBlob(next.blob, next.questionIndex);
  }, [processingQueue]);

  const averageScore = () => {
    if (scores.length === 0) return 0;
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  };


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

            <div className="mb-8 p-6 bg-primary/10 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">Average Depression Score</h2>
          <p className="text-3xl font-bold text-primary">
            {averageScore().toFixed(2)}
          </p>
        </div>
            
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

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Mic className="h-5 w-5 text-primary" />
              <span>Voice Recording</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-center min-h-[420px] flex flex-col justify-center">
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
                  <source src={recordedAudio} type="audio/webm" />
                  Your browser does not support the audio element.
                </audio>

                <div className="flex space-x-3 mt-2">
                  <Button 
                      variant="outline" 
                      onClick={() => {
                        setRecordedAudio(null);
                        setRecordedBlob(null);
                        setProcessingQueue((prev) =>
                          prev.filter(item => item.questionIndex !== currentQuestion)
                        );
                        setScores((prev) =>
                          prev.filter((_, idx) => idx !== currentQuestion)
                        );
                        setAnswers((prev) =>
                          prev.filter((_, idx) => idx !== currentQuestion)
                        );
                        setShowTranscript(false);
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
                  {processingQueue.length > 0 && (
                    <>
                      <p className="text-sm text-muted-foreground mb-2">
                        Remaining in queue: {processingQueue.length}
                      </p>
                      <Progress
                        value={((questions.length - processingQueue.length) / questions.length) * 100}
                        className="h-2"
                      />
                    </>
                  )}
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

                {transcript && (
                  <div className="mt-6 text-left p-4 rounded-lg bg-muted/30">
                    <div className="text-xs text-muted-foreground mb-1">Transcript</div>
                    <div className="text-sm">{transcript}</div>
                  </div>
                )}
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

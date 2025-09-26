import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  TrendingUp, 
  TrendingDown, 
  Mic, 
  Pause, 
  BookOpen, 
  Calendar,
  Download,
  Phone,
  Activity,
  Brain,
  AlertCircle,
  CheckCircle,
  Clock
} from "lucide-react";

export const UserDashboard = () => {
  // Mock data - in real app, this would come from API
  const riskScore = 35; // Low risk (0-40: Low, 41-70: Moderate, 71-100: High)
  const riskLevel = riskScore <= 40 ? "Low" : riskScore <= 70 ? "Moderate" : "High";
  const riskColor = riskScore <= 40 ? "text-success" : riskScore <= 70 ? "text-warning" : "text-destructive";

  const explainableIndicators = [
    { 
      icon: Mic, 
      label: "Speech Tone", 
      value: "Normal variation", 
      status: "positive",
      description: "Good emotional expression in voice"
    },
    { 
      icon: Pause, 
      label: "Speech Patterns", 
      value: "Some hesitation", 
      status: "neutral",
      description: "Slight increase in pauses detected"
    },
    { 
      icon: BookOpen, 
      label: "Word Choice", 
      value: "Positive sentiment", 
      status: "positive",
      description: "Generally optimistic language patterns"
    },
    { 
      icon: Activity, 
      label: "Energy Level", 
      value: "Moderate energy", 
      status: "neutral",
      description: "Voice energy within normal range"
    },
  ];

  const recentAnalyses = [
    { date: "2024-01-15", score: 28, trend: "down" },
    { date: "2024-01-10", score: 35, trend: "up" },
    { date: "2024-01-05", score: 32, trend: "stable" },
    { date: "2023-12-30", score: 40, trend: "up" },
  ];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-primary/10 to-healing-green/10 rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-2">Welcome back, Alex!</h1>
        <p className="text-muted-foreground">
          Your latest voice analysis was completed on January 15, 2024. 
          Keep tracking your mental wellness journey.
        </p>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Risk Score Card */}
        <div className="lg:col-span-1">
          <Card className="h-full">
            <CardHeader className="text-center">
              <CardTitle>Depression Risk Score</CardTitle>
              <CardDescription>
                Based on your latest voice analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <div className="relative w-32 h-32 mx-auto mb-4">
                <svg className="w-32 h-32 transform -rotate-90">
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="hsl(var(--muted))"
                    strokeWidth="8"
                    fill="none"
                  />
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke={`hsl(var(--${riskScore <= 40 ? 'success' : riskScore <= 70 ? 'warning' : 'destructive'}))`}
                    strokeWidth="8"
                    fill="none"
                    strokeLinecap="round"
                    strokeDasharray={`${(riskScore / 100) * 351.86} 351.86`}
                    className="transition-all duration-1000"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold">{riskScore}</span>
                  <span className="text-sm text-muted-foreground">/ 100</span>
                </div>
              </div>
              
              <Badge 
                variant={riskLevel === "Low" ? "default" : riskLevel === "Moderate" ? "secondary" : "destructive"}
                className="mb-4"
              >
                <div className="flex items-center space-x-1">
                  {riskLevel === "Low" ? (
                    <CheckCircle className="h-3 w-3" />
                  ) : riskLevel === "Moderate" ? (
                    <Clock className="h-3 w-3" />
                  ) : (
                    <AlertCircle className="h-3 w-3" />
                  )}
                  <span>{riskLevel} Risk</span>
                </div>
              </Badge>
              
              <p className="text-sm text-muted-foreground mb-4">
                {riskLevel === "Low" 
                  ? "Your mental wellness indicators are within healthy ranges."
                  : riskLevel === "Moderate"
                  ? "Some indicators suggest monitoring your mental wellness."
                  : "Consider reaching out for professional support."
                }
              </p>

              <div className="space-y-2">
                <Button className="w-full" size="sm">
                  <Calendar className="h-4 w-4 mr-2" />
                  Book Appointment
                </Button>
                <Button variant="outline" className="w-full" size="sm">
                  <Phone className="h-4 w-4 mr-2" />
                  Request Call
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Progress Chart */}
        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>Progress Over Time</CardTitle>
              <CardDescription>
                Your depression risk scores from recent analyses
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentAnalyses.map((analysis, index) => (
                  <div key={analysis.date} className="flex items-center space-x-4">
                    <div className="w-20 text-sm text-muted-foreground">
                      {new Date(analysis.date).toLocaleDateString('en-US', { 
                        month: 'short', 
                        day: 'numeric' 
                      })}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium">Score: {analysis.score}</span>
                        <div className="flex items-center space-x-1">
                          {analysis.trend === "up" ? (
                            <TrendingUp className="h-4 w-4 text-destructive" />
                          ) : analysis.trend === "down" ? (
                            <TrendingDown className="h-4 w-4 text-success" />
                          ) : (
                            <div className="h-4 w-4" />
                          )}
                        </div>
                      </div>
                      <Progress 
                        value={analysis.score} 
                        className="h-2"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Explainable AI Indicators */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>AI Analysis Breakdown</span>
          </CardTitle>
          <CardDescription>
            Understanding the factors that influenced your risk assessment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {explainableIndicators.map((indicator, index) => {
              const Icon = indicator.icon;
              const statusColor = 
                indicator.status === "positive" ? "text-success" : 
                indicator.status === "neutral" ? "text-warning" : 
                "text-destructive";
              
              return (
                <div key={index} className="p-4 rounded-lg border bg-card/50">
                  <div className="flex items-center space-x-3 mb-2">
                    <Icon className={`h-5 w-5 ${statusColor}`} />
                    <span className="font-medium text-sm">{indicator.label}</span>
                  </div>
                  <p className="text-sm font-semibold mb-1">{indicator.value}</p>
                  <p className="text-xs text-muted-foreground">{indicator.description}</p>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card className="hover:shadow-md transition-shadow cursor-pointer">
          <CardContent className="p-6 text-center">
            <Mic className="h-8 w-8 text-primary mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Take Voice Quiz</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Complete your next voice analysis session
            </p>
            <Button className="w-full">Start Quiz</Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow cursor-pointer">
          <CardContent className="p-6 text-center">
            <Download className="h-8 w-8 text-primary mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Download Report</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Get detailed analysis in PDF format
            </p>
            <Button variant="outline" className="w-full">Download PDF</Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow cursor-pointer">
          <CardContent className="p-6 text-center">
            <Calendar className="h-8 w-8 text-primary mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Schedule Check-in</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Book a session with a therapist
            </p>
            <Button variant="outline" className="w-full">View Calendar</Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
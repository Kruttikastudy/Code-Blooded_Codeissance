import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useState } from "react";

type TherapistReport = {
  id: string;
  name: string;
  specialization: string;
  userScore: number;
  therapistScore: number;
  aiScore: number;
  sessionSummary: string;
  keyFindings: string[];
  recommendations: string[];
};

const REPORTS: TherapistReport[] = [
  {
    id: "t1",
    name: "Dr. Aisha Sharma",
    specialization: "Clinical Psychologist",
    userScore: 78,
    therapistScore: 72,
    aiScore: 80,
    sessionSummary:
      "Discussed recent stressors and persistent low mood; explored coping strategies and routine building.",
    keyFindings: [
      "Negative cognitive bias in self-talk",
      "Sleep disturbances reported 4-5 nights/week",
      "Reduced engagement in previously enjoyable activities",
    ],
    recommendations: [
      "Daily thought journal with CBT reframing",
      "Consistent sleep routine; reduce screens before bed",
      "Schedule two enjoyable activities per week",
    ],
  },
  {
    id: "t2",
    name: "Dr. Rohan Mehta",
    specialization: "Counseling Psychologist",
    userScore: 70,
    therapistScore: 68,
    aiScore: 73,
    sessionSummary:
      "Focused on work-related stress and rumination; identified triggers and relaxation techniques.",
    keyFindings: [
      "Racing thoughts in evenings",
      "Work-life boundary issues",
      "Mild social withdrawal",
    ],
    recommendations: [
      "Evening wind-down routine (10 mins breathwork)",
      "Pomodoro with planned breaks",
      "Weekly social check-in with a friend",
    ],
  },
  {
    id: "t3",
    name: "Dr. Neha Kapoor",
    specialization: "Psychiatrist",
    userScore: 65,
    therapistScore: 62,
    aiScore: 67,
    sessionSummary:
      "Reviewed medication adherence and mood fluctuations; discussed follow-up plan.",
    keyFindings: [
      "Inconsistent routine",
      "Energy dips late afternoon",
      "Occasional anhedonia",
    ],
    recommendations: [
      "Structured daily schedule",
      "Hydration + short walk post-lunch",
      "Track mood in app daily",
    ],
  },
];

export const Reports = () => {
  const [selectedId, setSelectedId] = useState<string | null>(REPORTS[0]?.id ?? null);
  const selected = REPORTS.find(r => r.id === selectedId) ?? null;

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Reports</h1>
        <p className="text-muted-foreground">Your therapist consultations and detailed analyses</p>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        {REPORTS.map((r) => (
          <Card key={r.id} className={selectedId === r.id ? "ring-1 ring-primary" : ""}>
            <CardHeader>
              <CardTitle className="text-base">{r.name}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="text-sm text-muted-foreground">{r.specialization}</div>
              <div className="text-sm">Your Score: {r.userScore}</div>
              <div className="text-sm">Therapist Score: {r.therapistScore}</div>
              <Button variant="secondary" size="sm" onClick={() => setSelectedId(r.id)}>
                View detailed report
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      {selected && (
        <Card>
          <CardHeader>
            <CardTitle>Detailed Report — {selected.name}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid sm:grid-cols-3 gap-4">
              <div className="rounded-md border p-3">
                <div className="text-sm text-muted-foreground">Therapist Score</div>
                <div className="text-2xl font-semibold">{selected.therapistScore}</div>
              </div>
              <div className="rounded-md border p-3">
                <div className="text-sm text-muted-foreground">AI Analysis Score</div>
                <div className="text-2xl font-semibold">{selected.aiScore}</div>
              </div>
              <div className="rounded-md border p-3">
                <div className="text-sm text-muted-foreground">Your Latest Score</div>
                <div className="text-2xl font-semibold">{selected.userScore}</div>
              </div>
            </div>

            <div className="rounded-md border p-3">
              <div className="text-sm text-muted-foreground mb-1">Session Analysis</div>
              <p>{selected.sessionSummary}</p>
            </div>

            <div className="rounded-md border p-3">
              <div className="text-sm text-muted-foreground mb-1">Key Findings</div>
              <ul className="list-disc pl-6 space-y-1">
                {selected.keyFindings.map((k) => (
                  <li key={k}>{k}</li>
                ))}
              </ul>
            </div>

            <div className="rounded-md border p-3">
              <div className="text-sm text-muted-foreground mb-1">Recommendations</div>
              <ul className="list-disc pl-6 space-y-1">
                {selected.recommendations.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Reports;



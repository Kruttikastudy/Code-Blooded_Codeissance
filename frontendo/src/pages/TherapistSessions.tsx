import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, Plus } from "lucide-react";

const SESSIONS = [
  { id: 1, client: "Alex Johnson", date: "2025-09-10", summary: "Discussed coping strategies; assigned CBT journaling." },
  { id: 2, client: "Sarah Chen", date: "2025-09-09", summary: "Explored work stress; breathwork routine introduced." },
];

export const TherapistSessions = () => {
  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div className="bg-gradient-to-r from-healing-green/10 to-primary/10 rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-2">Sessions</h1>
        <p className="text-muted-foreground">Your recent session notes and summaries</p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Session Notes</CardTitle>
              <CardDescription>Keep track of client sessions</CardDescription>
            </div>
            <Button size="sm"><Plus className="h-4 w-4 mr-1" />New Note</Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {SESSIONS.map(s => (
              <div key={s.id} className="p-4 rounded-lg border bg-card/50">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">{s.client}</div>
                    <div className="text-xs text-muted-foreground">{new Date(s.date).toLocaleDateString()}</div>
                  </div>
                  <Button variant="ghost" size="sm"><FileText className="h-4 w-4 mr-1" />Open</Button>
                </div>
                <div className="text-sm text-muted-foreground mt-2">{s.summary}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TherapistSessions;



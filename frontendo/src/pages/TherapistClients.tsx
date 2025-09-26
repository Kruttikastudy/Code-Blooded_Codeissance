import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, FileText, Phone } from "lucide-react";

const CLIENTS = [
  { id: 1, name: "Alex Johnson", lastScore: 78, lastAnalysis: "2025-09-12" },
  { id: 2, name: "Sarah Chen", lastScore: 65, lastAnalysis: "2025-09-11" },
  { id: 3, name: "Michael Davis", lastScore: 28, lastAnalysis: "2025-09-10" },
];

export const TherapistClients = () => {
  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div className="bg-gradient-to-r from-healing-green/10 to-primary/10 rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-2">Clients</h1>
        <p className="text-muted-foreground">Overview of your clients and latest scores</p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Client List</CardTitle>
              <CardDescription>Recent analyses and quick actions</CardDescription>
            </div>
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Search clients..." className="pl-9" />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {CLIENTS.map(c => (
              <div key={c.id} className="p-4 rounded-lg border bg-card/50">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">{c.name}</div>
                    <div className="text-xs text-muted-foreground">Last analysis: {new Date(c.lastAnalysis).toLocaleDateString()}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="text-sm font-semibold">Score: {c.lastScore}</div>
                    <Button variant="ghost" size="sm"><FileText className="h-4 w-4 mr-1" />Notes</Button>
                    <Button variant="ghost" size="sm"><Phone className="h-4 w-4 mr-1" />Call</Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TherapistClients;



import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { useState } from "react";
import { format } from "date-fns";

const APPTS = [
  { time: "09:00", client: "Alex Johnson", type: "Follow-up" },
  { time: "11:30", client: "Sarah Chen", type: "Check-in" },
  { time: "14:00", client: "New Client", type: "Assessment" },
];

export const TherapistSchedule = () => {
  const [date, setDate] = useState<Date | undefined>(new Date());
  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div className="bg-gradient-to-r from-healing-green/10 to-primary/10 rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-2">Schedule</h1>
        <p className="text-muted-foreground">Manage your appointments</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Select Date</CardTitle>
            <CardDescription>{date ? format(date, "EEEE, dd MMM yyyy") : "No date"}</CardDescription>
          </CardHeader>
          <CardContent>
            <Calendar mode="single" selected={date} onSelect={setDate} className="rounded-md border" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Appointments</CardTitle>
            <CardDescription>For the selected date</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {APPTS.map((a, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                  <div className="text-sm font-medium text-primary min-w-[60px]">{a.time}</div>
                  <div className="flex-1">
                    <div className="font-medium text-sm">{a.client}</div>
                    <div className="text-xs text-muted-foreground">{a.type}</div>
                  </div>
                  <Button variant="ghost" size="sm">Join</Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default TherapistSchedule;



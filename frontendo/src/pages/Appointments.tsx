import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { format } from "date-fns";
import { VideoIcon } from "lucide-react";

const THERAPISTS = [
  { id: "t1", name: "Dr. Aisha Sharma", specialization: "Clinical Psychologist" },
  { id: "t2", name: "Dr. Rohan Mehta", specialization: "Counseling Psychologist" },
  { id: "t3", name: "Dr. Neha Kapoor", specialization: "Psychiatrist" },
];

const SLOTS = ["10:00", "11:30", "13:00", "15:00", "17:00", "19:00"];

export const Appointments = () => {
  const [date, setDate] = useState<Date | undefined>(new Date());
  const [therapistId, setTherapistId] = useState<string>(THERAPISTS[0].id);
  const [slot, setSlot] = useState<string>(SLOTS[0]);
  const [note, setNote] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [meetLink, setMeetLink] = useState<string>("");

  const selectedTherapist = THERAPISTS.find(t => t.id === therapistId)!;

  const handleBook = () => {
    const humanDate = date ? format(date, "EEEE, dd MMM yyyy") : "Select date";
    alert(`Booked ${selectedTherapist.name} on ${humanDate} at ${slot}.\nNote: ${note || "(none)"}`);
  };
  
  const handleJoinSession = async () => {
    if (!date) return;
    
    setIsLoading(true);
    try {
      // Hardcoded Google Meet link for demonstration purposes
      const hardcodedMeetLink = "https://meet.google.com/rec-peqo-ehr";
      
      // Set the meet link and open it in a new tab
      setMeetLink(hardcodedMeetLink);
      window.open(hardcodedMeetLink, "_blank");
      
      // Simulate a short delay for UI feedback
      setTimeout(() => {
        setIsLoading(false);
      }, 500);
    } catch (error) {
      console.error("Error opening meeting link:", error);
      alert("An error occurred. Please try again later.");
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Appointments</h1>
        <p className="text-muted-foreground">Book a session with a therapist</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Select a date</CardTitle>
          </CardHeader>
          <CardContent>
            <Calendar mode="single" selected={date} onSelect={setDate} className="rounded-md border" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Booking Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <Label>Therapist</Label>
                <Select value={therapistId} onValueChange={setTherapistId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select therapist" />
                  </SelectTrigger>
                  <SelectContent>
                    {THERAPISTS.map((t) => (
                      <SelectItem key={t.id} value={t.id}>
                        {t.name} — {t.specialization}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Time Slot</Label>
                <Select value={slot} onValueChange={setSlot}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select time" />
                  </SelectTrigger>
                  <SelectContent>
                    {SLOTS.map((s) => (
                      <SelectItem key={s} value={s}>{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <Label>Note for therapist (optional)</Label>
              <Input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Brief context you'd like to share…" />
            </div>

            <div className="flex flex-col gap-2">
              <Button className="w-full" onClick={handleBook} disabled={!date}>
                Confirm Booking
              </Button>
              
              <Button 
                className="w-full" 
                variant="outline" 
                onClick={handleJoinSession} 
                disabled={!date || isLoading}
              >
                {isLoading ? "Creating session..." : (
                  <>
                    <VideoIcon className="mr-2 h-4 w-4" />
                    Join Session
                  </>
                )}
              </Button>
            </div>

            <div className="text-sm text-muted-foreground">
              Selected: {date ? format(date, "dd MMM yyyy") : "No date"} at {slot}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Appointments;



public class Message {
    private final String sender;
    private final String text;
    private final User[] recipients;
    private final String timestamp;

    public Message(String givenUsername, String givenText, User[] givenRecipients, String givenTimestamp)
    {
        this.sender = givenUsername;
        this.text = givenText;
        this.recipients = givenRecipients;
        this.timestamp = givenTimestamp;
    }

    public User[] getRecipiets()
    {
        return this.recipients;
    }

    public String getSender() {
        return sender;
    }

    public String getText() {
        return text;
    }

    public String getTimestamp()
    {return this.timestamp;}
}

public class Message {
    private final String username;
    private final String text;

    public Message(String givenUsername, String givenText)
    {
        this.username = givenUsername;
        this.text = givenText;
    }


    public String getUsername() {
        return username;
    }

    public String getText() {
        return text;
    }
}

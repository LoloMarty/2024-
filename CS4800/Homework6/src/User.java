import java.util.*;
public class User {
    
    private ChatServer chatServer = new ChatServer();
    private String username;

    public User(String givenUsername)
    {
        this.username = givenUsername;

    }

    public void sendMessage(Message givenMessage)
    {
        chatServer.sendMessage(this.username, givenMessage);
    }
    public void undo()
    {
        this.chatServer.undo();
    }

}

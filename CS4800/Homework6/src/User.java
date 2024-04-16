import java.util.Deque;
import java.util.LinkedList;

public class User {
    private final ChatServer server;
    private final String username;
    private LinkedList<Message> percievedChat;

    public User (String givenUsername)
    {
        this.username = givenUsername;
        this.server = ChatServer.getChatServerInstance();
    }

    public void writeToChat(String givenMessage)
    {
        server.write(new Message(this.username, givenMessage));
    }

    public void undoLastMessage()
    {
        this.server.undoLastMessage();
    }

    private LinkedList<Message> duplicateLinkedList(LinkedList<Message> originalList) {
        LinkedList<Message > duplicateList = new LinkedList<>();

        duplicateList.addAll(originalList);

        return duplicateList;
    }

    public void updatePercievedChat(LinkedList<Message> givenChat)
    {
        this.percievedChat = duplicateLinkedList(givenChat);
    }

    public void printPercievedChat()
    {
        Deque<Message> chat = this.percievedChat;
        Message firstMessageNode = chat.peekLast();
        Message currentMessageNode = chat.pop();

        System.out.println("\n*** " + this.username + " sees the chat as: ***");

        while(!(firstMessageNode.equals(currentMessageNode)))
        {
            System.out.println(currentMessageNode.getUsername() + " said:");
            System.out.println("\t"+currentMessageNode.getText());
            chat.addLast(currentMessageNode);
            currentMessageNode = chat.pop();
        }

        System.out.println(currentMessageNode.getUsername() + " said:");
        System.out.println("\t"+currentMessageNode.getText());
        chat.addLast(currentMessageNode);
    }
}

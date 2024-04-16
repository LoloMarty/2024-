import java.util.Deque;
import java.util.LinkedList;

public class ChatServer {
    private static ChatServer instance;
    private static ChatHistory historyKeeper;
    private static LinkedList<User> listOfUsers;

    public ChatServer()
    {
        historyKeeper = ChatHistory.getInstance();
        listOfUsers = new LinkedList<>();
    }

    public static ChatServer getChatServerInstance()
    {
        if (instance == null)
        {
            instance = new ChatServer();
        }

        return instance;
    }

    private void updateAllUserChats()
    {
        LinkedList<User> users = this.duplicateLinkedList(listOfUsers);
        User firstUser = users.peekLast();
        User currentUser = currentUser = users.pop();

        System.out.println();

        while(!(firstUser.equals(currentUser)))
        {
            currentUser.updatePercievedChat(this.getChat());
            users.addLast(currentUser);
            currentUser = users.pop();
        }

        currentUser.updatePercievedChat(this.getChat());
    }

    public void write(Message givenMessage)
    {
        historyKeeper.addMessage(givenMessage);
        updateAllUserChats();
    }

    public void undoLastMessage()
    {
        historyKeeper.undo();
        updateAllUserChats();
    }

    public LinkedList<Message> getChat()
    {
        return historyKeeper.getChatState();
    }

    public User addUser(String givenUsername)
    {
        User newUser = new User(givenUsername);
        listOfUsers.add(newUser);

        return newUser;
    }

    public void printChatToConsole()
    {
        Deque<Message> chat = this.getChat();
        Message firstMessageNode = chat.peekLast();
        Message currentMessageNode = currentMessageNode = chat.pop();

        System.out.println();

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

    private LinkedList<User> duplicateLinkedList(LinkedList<User> originalList) {
        LinkedList<User> duplicateList = new LinkedList<>();

        duplicateList.addAll(originalList);

        return duplicateList;
    }

}

import java.util.LinkedList;

public class ChatServer {
    private static ChatServer instance;
    private static LinkedList<User> listOfUsers;
    private static LinkedList<User> listOfBlockedUsers;

    public ChatServer()
    {
        listOfUsers = new LinkedList<>();
        listOfBlockedUsers = new LinkedList<>();
    }

    public static ChatServer getChatServerInstance()
    {
        if (instance == null)
        {
            instance = new ChatServer();
        }

        return instance;
    }

    public void write(Message givenMessage)
    {
        if (!isSenderBlocked(givenMessage))
        {
            for(User user : givenMessage.getRecipiets())
            {
                user.updatePercievedChat(givenMessage);
            }
        }
    }

    public User registerUser(String givenUsername)
    {
        User newUser = new User(givenUsername);
        listOfUsers.add(newUser);

        return newUser;
    }

    public void unregisterUser(String givenUsername)
    {
        LinkedList<User> users = listOfUsers;
        User currentUser = users.pop();

        System.out.println();

        while(!(givenUsername.equals(currentUser.getUsername())))
        {
            users.addLast(currentUser);
            currentUser = users.pop();
        }

        currentUser = users.pop();
    }

    public void addBlockedUser(User userToBlock)
    {
        listOfBlockedUsers.add(userToBlock);
    }

    public boolean isSenderBlocked(Message messageToEvaluate)
    {
        boolean returnValue = false;

        for(User user : listOfBlockedUsers)
        {
            if (user.getUsername().equals(messageToEvaluate.getSender())) {
                returnValue = true;
                break;
            }
        }

        return returnValue;
    }

    public void sendUndoRequest(User userRequesting)
    {
        for(User user : listOfUsers)
        {
            user.processUndoLastMessage(userRequesting);
        }
    }
}

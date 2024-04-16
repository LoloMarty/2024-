public class DriverProgram {
    public static void main(String[] args) {
        ChatServer server = new ChatServer();

        User user1 = server.registerUser("Marvin");
        User user2 = server.registerUser("Sean");
        User user3 = server.registerUser("Nima");

        server.addBlockedUser(user2);

        user1.writeToChat("Hello guys!", new User[]{user2, user3});
        user2.writeToChat("Hi Marvin", new User[]{user1, user3});
        user3.writeToChat("Why are you guys not in class?", new User[]{user1, user2});

        user1.printPercievedChat();
        user2.printPercievedChat();
        user3.printPercievedChat();

        user1.requestUndoLastMessage();

        System.out.println();

        user1.printPercievedChat();
        user2.printPercievedChat();
        user3.printPercievedChat();
    }
}
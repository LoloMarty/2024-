public class DriverProgram {
    public static void main(String[] args) {
        ChatServer server = new ChatServer();

        User user1 = server.addUser("Marvin");
        User user2 = server.addUser("Sean");
        User user3 = server.addUser("Nima");

        user1.writeToChat("Hello guys!");
        user2.writeToChat("Hi Marvin");
        user3.writeToChat("Why are you guys not in class?");

        user1.printPercievedChat();
        user2.printPercievedChat();
        user3.printPercievedChat();



    }
}
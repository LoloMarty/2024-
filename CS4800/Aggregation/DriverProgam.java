package CS4800.Aggregation;

public class DriverProgam {

    public static void main(String[] args) {

        Instructor instructor = new Instructor("Nima", "Davarpanah", "3-2636");
        Instructor instructor2 = new Instructor("Hao", "Ji", "3-1535");
        Textbook textbook = new Textbook("Clean Code", "Robert C. Martin", "Lighthouse Books");
        Textbook textbook2 = new Textbook("To Kill a Mockingbird", "Harper Lee", "Penguin Books");

        Course course = new Course("Software Engineering", instructor, instructor2, textbook, textbook2);

        course.printClassInformation();

    }
}

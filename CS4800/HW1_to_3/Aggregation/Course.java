package CS4800.Aggregation;

public class Course {
    private String courseName;
    private Instructor instructor;
    private Instructor instructor2;
    private Textbook textbook;
    private Textbook textbook2;

    public Instructor getInstructor2() {
        return instructor2;
    }

    public void setInstructor2(Instructor instructor2) {
        this.instructor2 = instructor2;
    }

    public Textbook getTextbook2() {
        return textbook2;
    }

    public void setTextbook2(Textbook textbook2) {
        this.textbook2 = textbook2;
    }

    public String getCourseName() {
        return courseName;
    }

    public void setCourseName(String name) {
        this.courseName = name;
    }

    public Instructor getInstructor() {
        return instructor;
    }

    public void setInstructor(Instructor instructor) {
        this.instructor = instructor;
    }

    public Textbook getTextbook() {
        return textbook;
    }

    public void setTextbook(Textbook textbook) {
        this.textbook = textbook;
    }

    public void printClassInformation() {
        System.out.printf(
                "\nCourse Name: %s\nInstructor 1 Name: %s %s\nInstructor 1 Office: %s\nInstructor 2 Name: %s %s\nInstructor 2 Office: %s\n"
                        +
                        "Textbook 1 Title: %s\nTextbook 1 Author: %s\nTextbook 2 Title: %s\nTextbook 2 Author: %s",
                this.getCourseName(), this.instructor.getFirstName(), this.instructor.getLastName(),
                this.instructor.getOfficeNumber(),
                this.getInstructor2().getFirstName(), this.instructor2.getLastName(),
                this.instructor2.getOfficeNumber(), this.textbook.getTitle(),
                this.textbook.getAuthor(), this.textbook2.getTitle(), this.textbook2.getAuthor());
    }

    public Course(String givenCourseName, Instructor givenInstructor1, Instructor givenInstructor2,
            Textbook givenTextbook1, Textbook givenTextbook2) {

        this.setCourseName(givenCourseName);
        this.setInstructor(givenInstructor1);
        this.setInstructor2(givenInstructor2);
        this.setTextbook(givenTextbook1);
        this.setTextbook2(givenTextbook2);
    }

}
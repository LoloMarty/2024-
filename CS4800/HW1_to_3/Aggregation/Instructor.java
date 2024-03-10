package CS4800.Aggregation;

public class Instructor {
    private String firstName;
    private String lastName;
    private String officeNumber;

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getOfficeNumber() {
        return officeNumber;
    }

    public void setOfficeNumber(String officeNumber) {
        this.officeNumber = officeNumber;
    }

    public Instructor(String givenFirstName, String givenLastName, String givenOfficeNumber) {
        this.setFirstName(givenFirstName);
        this.setLastName(givenLastName);
        this.setOfficeNumber(givenOfficeNumber);
    }
}

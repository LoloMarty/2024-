package CS4800.Composition;

public class File {
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void printNameOfFile() {
        System.out.printf("File Name: %s\n", this.getName());
    }

    public File(String givenName) {
        this.setName(givenName);
    }
}

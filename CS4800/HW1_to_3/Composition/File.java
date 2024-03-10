package CS4800.Composition;

public class File {
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    private void printIndents(int numberOfPrintedIndents) {
        for (int indent = 0; indent < numberOfPrintedIndents; indent++) {
            System.out.printf("   ");
        }
    }

    public void printNameOfFile(int numberOfPrintedIndents) {
        printIndents(numberOfPrintedIndents);
        System.out.printf("[File] %s\n", this.getName());
    }

    public File(String givenName) {
        this.setName(givenName);
    }
}

import java.util.ArrayList;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;

public class CharString {
    ArrayList<Character> string;
    String fileName;

    public CharString() {
        string = new ArrayList<Character>();
        fileName = "document.txt"; // Name of the file to write to
    }

    public void save(String givenCharacter, String givenFont, String givenColor, int givenSize) {
        string.add(new Character(givenCharacter, givenFont, givenColor, givenSize));

        try {
            FileWriter writer = new FileWriter(fileName);
            writer.write(this.buildString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String buildString() {
        String builtString = "";
        for (Character character : this.string) {
            builtString += character.getHeldCharacter();
        }

        return builtString;
    }

    public void load() {
        try {
            FileReader reader = new FileReader(fileName);
            BufferedReader bufferedReader = new BufferedReader(reader);

            String line;
            while ((line = bufferedReader.readLine()) != null) {
                System.out.println("\n" + line + "\n");
            }

            bufferedReader.close();
        } catch (IOException e) {
            System.out.println("An error occurred while reading the file.");
            e.printStackTrace();
        }

        for (Character character : this.string) {
            System.out.printf("Char: %s\n", character.getHeldCharacter());
            character.printCharacaterAttributes();
            System.out.println("\n");
        }
    }
}

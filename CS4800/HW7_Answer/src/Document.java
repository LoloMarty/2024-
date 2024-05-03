import java.util.List;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

// Document object that stores a list of characters and allows the user to create, edit, and save documents
public class Document {

    private final List<Character> characters = new ArrayList<>();
    private final CharacterPropertiesFactory propertiesFactory = new CharacterPropertiesFactory();

    public void addCharacter(char c, String font, String color, int size) {
        CharacterProperties flyweightCharacterProperties = propertiesFactory.setAndRetrieveFlyweightCharacterProperties(font, color, size);
        Character character = new Character(c, flyweightCharacterProperties); //character property is really a flyweight object.
        characters.add(character);
    }

    public void editCharacterProperties(int index, String font, String color, int size) {
        Character character = characters.get(index);
        CharacterProperties flyweightCharacterProperties = propertiesFactory.setAndRetrieveFlyweightCharacterProperties(font, color, size);
        character.setCharacterProperty(flyweightCharacterProperties);
    }

    public void saveToFile(String filename) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
        for (Character character : characters) {
            writer.write(character.getCharacter() + "," + character.getFont() + "," + character.getColor() + "," + character.getSize() + "\n");
        }
        writer.close();
        System.out.println("the number of flyweight objects is: " + propertiesFactory.sizeOfMap());
    }

    public void loadFromFile(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            char c = parts[0].charAt(0);
            String font = parts[1];
            String color = parts[2];
            int size = Integer.parseInt(parts[3]);
            this.addCharacter(c, font, color, size);
        }
    }
}

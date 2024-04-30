public class Disk {
    static CharString document;

    public static CharString getDocument() {
        if (document == null) {
            document = new CharString();
        }

        return document;
    }
}

package CS4800.Composition;

public class Folder {
    private Folder[] subFolders;
    private File[] files;
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Folder[] getSubFolders() {
        return subFolders;
    }

    public void setSubFolders(Folder[] subFolders) {
        this.subFolders = subFolders;
    }

    public File[] getFiles() {
        return files;
    }

    public void setFiles(File[] files) {
        this.files = files;
    }

    public void addFile(File givenFile) {
        File[] newFilesGroup = fileGroupSizePlusOne();
        newFilesGroup = this.copyFiles(this.getFiles());
        this.setFiles(newFilesGroup);
    }

    public void addFolder(Folder givenFolder) {
        Folder[] newSubFoldersGroup = subFolderGroupSizePlusOne();
        newSubFoldersGroup = this.copyFolders(this.getSubFolders());
        this.setSubFolders(newSubFoldersGroup);
    }

    private Folder[] subFolderGroupSizePlusOne() {
        return new Folder[this.subFolders.length + 1];
    }

    private File[] fileGroupSizePlusOne() {
        return new File[this.files.length + 1];
    }

    private Folder[] copyFolders(Folder[] foldersToCopy) {
        Folder[] newFoldersGroupToReturn = new Folder[foldersToCopy.length];

        for (int file = 0; file < foldersToCopy.length; file++) {
            newFoldersGroupToReturn[file] = foldersToCopy[file];
        }

        return newFoldersGroupToReturn;
    }

    private File[] copyFiles(File[] filesToCopy) {
        File[] newFilesGroupToReturn = new File[filesToCopy.length];

        for (int file = 0; file < filesToCopy.length; file++) {
            newFilesGroupToReturn[file] = filesToCopy[file];
        }

        return newFilesGroupToReturn;
    }

    public Folder(Folder[] givenSubFolders, String givenName) {
        this.setName(givenName);
        this.subFolders = new Folder[givenSubFolders.length];
        this.files = new File[0];
        this.setSubFolders(this.copyFolders(givenSubFolders));
    }

    public Folder(File[] givenFiles, String givenName) {
        this.setName(givenName);
        this.subFolders = new Folder[0];
        this.files = new File[givenFiles.length];
        this.setFiles(this.copyFiles(givenFiles));
    }

    public Folder(Folder[] givenSubFolders, File[] givenFiles, String givenName) {
        this.setName(givenName);
        this.subFolders = new Folder[givenSubFolders.length];
        this.files = new File[givenFiles.length];

        this.setSubFolders(this.copyFolders(givenSubFolders));
        this.setFiles(this.copyFiles(givenFiles));
    }

    public Folder (Folder givenSubFolder, String givenName)
    {
        this.setName(givenName);
        this.subFolders = new Folder[0];
        this.files = new File[0];
        this.addFolder(givenSubFolder);
    }

    public Folder (File givenFile, String givenName)
    {
        this.setName(givenName);
        this.subFolders = new Folder[0];
        this.files = new File[0];
        this.addFile(givenFile);
    }

    public Folder (String givenName)
    {
        this.setName(givenName);
        this.subFolders = new Folder[0];
        this.files = new File[0];
    }

    private void printIndents(int numberOfPrintedIndents) {
        for (int indent = 0; indent < numberOfPrintedIndents; indent++) {
            System.out.printf("   ");
        }
    }

    private void printNamesOfAllFilesInCurrentFolder(int numberOfPrintedIndents) {
        for (int file = 0; file < this.files.length; file++) {
            printIndents(numberOfPrintedIndents);
            this.files[file].printNameOfFile(numberOfPrintedIndents+1);
        }
    }

    private void printAllSubFoldersInCurrentFolder(int numberOfPrintedIndents) {
        for (int folder = 0; folder < this.subFolders.length; folder++) {
            printIndents(numberOfPrintedIndents);
            this.subFolders[folder].printEntireDirectory_helper(numberOfPrintedIndents+1);
        }
    }

    public void printEntireDirectory_helper(int numberOfPrintedIndents) {
        printIndents(numberOfPrintedIndents);
        System.out.printf("[Folder] %s\n", this.getName());

        printNamesOfAllFilesInCurrentFolder(numberOfPrintedIndents);
        printAllSubFoldersInCurrentFolder(numberOfPrintedIndents);
    }

    public void printEntireDirectory() {
        System.out.printf("[Folder] %s", this.getName());
        printNamesOfAllFilesInCurrentFolder(0);
        printAllSubFoldersInCurrentFolder(0);
    }
}
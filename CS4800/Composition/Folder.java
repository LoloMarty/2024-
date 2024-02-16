package CS4800.Composition;

public class Folder {
    private Folder[] subFolders;
    private File[] files;

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
        File[] newFilesGroup = incrementFileGroupSize();
        newFilesGroup = this.copyFiles(this.getFiles());
        this.setFiles(newFilesGroup);
    }

    public void addFolder(Folder givenFolder) {
        Folder[] newSubFoldersGroup = incrementSubFolderGroupSize();
        newSubFoldersGroup = this.copyFolders(this.getSubFolders());
        this.setSubFolders(newSubFoldersGroup);
    }

    private Folder[] incrementSubFolderGroupSize() {
        return new Folder[this.subFolders.length + 1];
    }

    private File[] incrementFileGroupSize() {
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

    public Folder(Folder[] givenSubFolders) {
        this.subFolders = new Folder[givenSubFolders.length];
        this.files = new File[0];
        this.setSubFolders(this.copyFolders(givenSubFolders));
    }

    public Folder(File[] givenFiles) {
        this.subFolders = new Folder[0];
        this.files = new File[givenFiles.length];
        this.setFiles(this.copyFiles(givenFiles));
    }

    public Folder(Folder[] givenSubFolders, File[] givenFiles) {
        this.subFolders = new Folder[givenSubFolders.length];
        this.files = new File[givenFiles.length];

        this.setSubFolders(this.copyFolders(givenSubFolders));
        this.setFiles(this.copyFiles(givenFiles));
    }

    private void printIndents(int numberOfPrintedIndents) {
        for (int indent = 0; indent < numberOfPrintedIndents; indent++) {
            System.out.printf("\t");
        }
    }

    private void printNamesOfAllFilesInCurrentFolder(int numberOfPrintedIndents) {
        for (int file = 0; file < this.files.length; file++) {
            printIndents(numberOfPrintedIndents);
            this.files[file].printNameOfFile();
        }
    }

    private void printAllSubFoldersInCurrentFolder(int numberOfPrintedIndents) {
        for (int folder = 0; folder < this.subFolders.length; folder++) {
            printIndents(numberOfPrintedIndents);
            this.subFolders[folder].printEntireDirectory_helper(numberOfPrintedIndents);
        }
    }

    public void printEntireDirectory_helper(int numberOfPrintedIndents) {
        printIndents(numberOfPrintedIndents);
        System.out.printf("Sub Folder");

        printNamesOfAllFilesInCurrentFolder(numberOfPrintedIndents);
        printAllSubFoldersInCurrentFolder(numberOfPrintedIndents);
    }

    public void printEntireDirectory() {
        printNamesOfAllFilesInCurrentFolder(0);
        printAllSubFoldersInCurrentFolder(0);
    }
}
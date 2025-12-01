# Social Network Analytics - Beginner's Guide

## What is this tool?

Social Network Analytics is a web application that helps you understand connections and relationships in social media data. You upload your data (like tweets, posts, or comments), and the tool creates a visual map showing who mentions whom, what topics are discussed, and how everything connects together.

**Think of it like:** A map that shows how ideas, people, and topics are connected in your social media data.

---

## Table of Contents

1. [Opening the Application](#opening-the-application)
2. [Step 1: Upload Your Data](#step-1-upload-your-data)
3. [Step 2: Choose What to Extract](#step-2-choose-what-to-extract)
4. [Step 3: Basic Settings](#step-3-basic-settings)
5. [Step 4: Run the Analysis](#step-4-run-the-analysis)
6. [Step 5: See Your Results](#step-5-see-your-results)
7. [Step 6: Download Your Network](#step-6-download-your-network)
8. [Common Questions](#common-questions)
9. [Troubleshooting](#troubleshooting)

---

## Opening the Application

Someone (like your IT department or researcher) will give you a web address (URL) to open in your browser. It might look like:
- `http://localhost:8501` (if running on your computer)
- Or a web address like `http://your-server.com:8501`

Just click the link and the application will open in your web browser (Chrome, Firefox, Safari, etc.).

**That's it!** You don't need to install anything or write any code.

---

## Step 1: Upload Your Data

### What kind of file do I need?

You need a spreadsheet file (usually called CSV) or a special text file (NDJSON).

**Most commonly:** An Excel file saved as CSV (Comma Separated Values)

### What should be in my file?

Your file needs at least two columns:
1. **Who wrote it** - usernames, names, or IDs of people
2. **What they wrote** - the actual text of posts, tweets, or comments

**Example:**

| Username | Post Text |
|----------|-----------|
| john_smith | "I love this product! #awesome" |
| mary_jane | "Thanks @john_smith for the recommendation!" |
| bob_jones | "Just visited Paris, amazing city!" |

### How do I upload?

1. Look for the section that says **"Upload Data"**
2. Click the **"Browse files"** or **"Choose a file"** button
3. Find your CSV file on your computer
4. Click **"Open"**

**You'll know it worked when:** You see a preview of your data showing the first few rows.

### Check your preview

After uploading, you'll see:
- **A table** showing the first 10 rows of your data
- **File information** like size and number of rows
- This helps you make sure you uploaded the right file!

---

## Step 2: Choose What to Extract

This is the most important decision! You're telling the tool what to look for in your text.

### Which column has the authors/users?

Find the dropdown menu labeled **"Author Column"** and select which column contains the usernames or author names.

**Common names for this column:**
- username
- author
- user
- name
- unique_id

**The app usually guesses this correctly!** But check to make sure.

### Which column has the text?

Find the dropdown menu labeled **"Text Column"** and select which column has the posts, tweets, or comments.

**Common names for this column:**
- text
- content
- message
- post
- body

**Again, the app usually guesses correctly.**

### What do you want to find in the text?

This is where you choose what the tool will extract. Look for the **"Choose Extraction Method"** menu on the left side.

#### Option 1: Named Entities (Default - Best for News & General Content)

**Choose this if:** Your posts mention real people, places, or organizations

**What it finds:**
- Names of people (like "Joe Biden", "Taylor Swift")
- Places (like "New York", "France", "Tokyo")
- Organizations (like "Google", "United Nations", "NASA")

**Good for:**
- News articles
- Political discussions
- Travel posts
- Business content

**Example:**
- Text: "Apple announced a new iPhone in California"
- Finds: Apple (organization), California (place)

#### Option 2: Hashtags (Best for Twitter/Instagram)

**Choose this if:** Your data has hashtags and you want to see which topics are popular

**What it finds:**
- Anything with a # symbol (like #politics, #love, #travel)

**Good for:**
- Twitter/X posts
- Instagram posts
- TikTok captions

**Example:**
- Text: "Beautiful sunset! #nature #photography #beach"
- Finds: nature, photography, beach

#### Option 3: Mentions (Best for Twitter/X Conversations)

**Choose this if:** You want to see who mentions whom

**What it finds:**
- Usernames mentioned with @ symbol (like @username)

**Good for:**
- Twitter/X conversations
- Instagram comments
- Finding influential users

**Example:**
- Text: "Great work @jane_doe and @bob_smith!"
- Finds: jane_doe, bob_smith

#### Option 4: Websites (Best for Link Sharing)

**Choose this if:** You want to see which websites are being shared

**What it finds:**
- Website addresses (like nytimes.com, youtube.com)

**Good for:**
- News sharing
- Link analysis
- Source tracking

**Example:**
- Text: "Check this out: https://www.bbc.com/news/article"
- Finds: bbc.com

#### Option 5: Keywords (Best for Finding Topics)

**Choose this if:** You want to find the main topics and themes

**What it finds:**
- Important words and phrases (automatically detected)
- Filters out common words like "the", "and", "is"

**Good for:**
- Topic discovery
- Content analysis
- Understanding what people talk about

**Example:**
- Text: "Climate change is causing severe weather patterns"
- Finds: climate change, severe weather patterns

#### Option 6: Exact Text (Advanced)

**Choose this if:** Your text column already has the exact categories you want

**Most people don't use this option** - it's for special cases.

---

## Step 3: Basic Settings

You can usually **skip this section** and use the default settings! But here's what they mean if you want to adjust:

### For Named Entities (if you chose that):

**Which types of things to find:**
- Check the boxes for what you want:
  - ✓ Persons (people's names)
  - ✓ Locations (places)
  - ✓ Organizations (companies, groups)

**Confidence:**
- This slider controls how "sure" the tool needs to be
- **Higher** = More accurate, but finds fewer things
- **Lower** = Finds more things, but may make mistakes
- **Default (85%) is usually good**

### For Keywords (if you chose that):

**How many keywords to find:**
- Minimum: At least this many keywords per person
- Maximum: No more than this many keywords per person
- **Default (5-20) is usually good**

**Language:**
- Choose the language of your text
- English is default
- Also supports Danish, Spanish, French, and others

### Other Settings (Usually Don't Need to Change):

**Chunk Size:**
- How many rows to process at once
- **Don't change this unless the app is slow or crashes**

---

## Step 4: Run the Analysis

### Ready to go!

Once you've:
1. ✓ Uploaded your file
2. ✓ Selected author and text columns
3. ✓ Chosen what to extract
4. ✓ Adjusted settings (or kept defaults)

### Click the big button!

Look for the blue button that says **"Start Processing"** or **"🚀 Start Processing"**

**Click it!**

### What happens next?

You'll see:
- A progress bar showing how much is done
- Messages like "Processing chunk 1 of 5..."
- Estimated time remaining

**How long does it take?**
- Small files (1,000 rows): 10-30 seconds
- Medium files (10,000 rows): 1-5 minutes
- Large files (100,000 rows): 10-30 minutes

**What to do while waiting:**
- Nothing! Just wait for it to finish
- Don't close the browser window
- Don't click "Start Processing" again

### When it's done

You'll see a success message with numbers like:
- "✅ Processing Complete!"
- "Posts processed: 5,000"
- "Entities extracted: 1,234"

**Now scroll down to see your results!**

---

## Step 5: See Your Results

### The Numbers

At the top, you'll see some statistics:

**Total Nodes:**
- This is everyone and everything in your network
- Authors (people who posted) + Things they mentioned

**Total Edges:**
- This is the connections between them
- "John mentions Paris" = 1 edge

**Authors:**
- How many unique people posted

**Entities:**
- How many unique things were found (people, places, hashtags, etc.)

**Don't worry too much about these numbers** - the visualization is more interesting!

### Top Mentioned Items

Below the numbers, you'll see a table showing:
- What was mentioned most often
- How many times it appeared
- What type it is (person, place, hashtag, etc.)

**Example:**

| Entity | Mentions | Type |
|--------|----------|------|
| New York | 45 | Location |
| climate change | 32 | Keyword |
| Biden | 28 | Person |

This tells you what your data is mostly about!

### The Network Visualization

This is the fun part! You'll see a **web of connected dots**.

**What am I looking at?**

- **Dots (Nodes)** = People or things
- **Lines (Edges)** = Connections
- **Colors**:
  - Blue dots = Authors (people who posted)
  - Orange dots = People mentioned
  - Green dots = Places
  - Red dots = Organizations
  - Other colors = Other types

**Size matters:**
- **Big dots** = Very connected, mentioned a lot
- **Small dots** = Less connected, mentioned rarely

**How to explore:**

1. **Hover your mouse over a dot** = See details about it
2. **Scroll your mouse wheel** = Zoom in and out
3. **Click and drag** = Move around the map
4. **Use the play button** = Watch the network rearrange itself

**What to look for:**

- **Clusters** = Groups of dots close together (related topics/communities)
- **Central dots** = Big dots in the middle (important people/topics)
- **Connections** = Who's connected to whom

### Understanding Your Network

**Dense cluster in the middle:**
- This is the main conversation
- Most connected people and topics

**Isolated dots on the edges:**
- Less connected
- Might be off-topic or one-time mentions

**Multiple clusters:**
- Different sub-topics or communities
- Each cluster is a different conversation

### Filter to Main Network

See the checkbox **"Giant Component Only"**?

**Check this box** to show only the biggest, most connected part of your network. This helps you focus on the main conversation.

---

## Step 6: Download Your Network

Ready to save your results? Scroll down to **"Download Results"**.

### Which file should I download?

**For most people:**

**Download GEXF** (the first big download button)
- This is the main file you want
- Used with a program called **Gephi** (free network visualization software)
- Best for creating professional-looking network diagrams

**To use it:**
1. Download the GEXF file
2. Go to https://gephi.org and download Gephi (free)
3. Open Gephi
4. File → Open → Select your GEXF file
5. Create beautiful network visualizations!

**Other files you might want:**

**Edge List CSV:**
- Opens in Excel
- Simple table showing: who → mentioned → what
- Good for basic analysis

**Statistics JSON:**
- Contains all the numbers
- For advanced users who want to analyze further

### Saving for later

Once you download the files, **save them somewhere safe!**

You can:
- Reopen them in Gephi anytime
- Share them with colleagues
- Include them in reports

**Tip:** Create a folder for your project and save all files there.

---

## Common Questions

### Q: I uploaded my file but nothing happens

**Check:**
- Is your file a CSV or NDJSON file?
- Does it have at least two columns (author and text)?
- Are the columns selected correctly?

**Try:**
- Refresh the page and upload again
- Make sure the file isn't corrupted

### Q: The processing is taking forever

**This is normal if:**
- Your file is very large (10,000+ rows)
- You chose Named Entities (this is the slowest method)

**To speed it up:**
- Try a smaller sample of your data first
- Choose a simpler method (like Hashtags or Mentions)

### Q: I got an error message

**Common fixes:**
- Refresh the page and try again
- Check that your columns are selected correctly
- Make sure your text column actually has text in it
- Try a different extraction method

### Q: The network looks like a mess

**This is normal!** Real social networks are messy.

**Try:**
- Check the "Giant Component Only" box
- Download to Gephi for better layout options
- Zoom in on specific clusters
- Filter to show only top items

### Q: I don't see any results

**Possible reasons:**
- Your text column is empty
- The extraction method didn't find anything
- Confidence is set too high (for Named Entities)

**Solutions:**
- Check your data preview - is there text?
- Try a different extraction method
- Lower the confidence slider (Named Entities)

### Q: Can I process the same data again?

**Yes!** You can:
- Try different extraction methods
- Adjust settings
- Process as many times as you want

The app is smart and remembers previous results, making it faster the second time.

### Q: Is my data safe?

**Yes!**
- Your data stays on the computer/server running the app
- Nothing is sent to the internet
- Only you can see your results

### Q: Do I need to install anything?

**No!**
- It works in your web browser
- No downloads needed
- No software to install

### Q: Can I save my work?

**Yes!**
- Download the files (Step 6)
- Save them on your computer
- You can reopen them anytime

### Q: What if I close the browser?

**Your results will disappear.**
- Make sure to download files before closing
- You can always upload and process again

### Q: Can I edit the visualization?

**In the app:** Limited editing (zoom, filter)

**For full editing:**
- Download GEXF file
- Open in Gephi
- Full control over colors, layout, labels, etc.

---

## Troubleshooting

### Problem: "Error uploading file"

**Solution:**
- Make sure it's a CSV file
- Try opening it in Excel first to check it's valid
- Check the file isn't too large (over 100MB might have issues)

### Problem: "No entities found"

**Solution:**
- Check your text column has actual text
- Try a different extraction method
- For Named Entities: lower the confidence slider

### Problem: "App is very slow"

**Solution:**
- Close other browser tabs
- Try a smaller file first (sample your data)
- Use a simpler extraction method (Hashtags instead of Named Entities)

### Problem: "Visualization won't show"

**Solution:**
- Refresh the page
- Try checking "Giant Component Only"
- The network might be too large - it shows top 500 nodes for large networks

### Problem: "Download button not working"

**Solution:**
- Check your browser allows downloads
- Try a different browser
- Right-click the button and "Save Link As"

### Problem: "Results don't make sense"

**Solution:**
- Check you selected the correct columns
- Verify the extraction method matches your data type
- Look at the data preview - is it what you expected?

### Problem: "Can't find the results"

**Solution:**
- Scroll down - results appear below the "Start Processing" button
- Make sure processing finished (look for "✅ Processing Complete")
- Check for error messages in red

---

## Need More Help?

**If you're stuck:**
1. Try refreshing the page and starting over
2. Test with a very small file first (just 100 rows)
3. Try the simplest extraction method (Hashtags or Mentions)
4. Ask whoever set up the app for help

**Things to tell them:**
- What extraction method you chose
- Any error messages you see
- How many rows your file has
- What happened when you clicked "Start Processing"

---

## Quick Start Checklist

Follow this simple checklist for success:

- [ ] Open the web app in your browser
- [ ] Upload your CSV file
- [ ] Check the data preview looks correct
- [ ] Select the author column (who wrote it)
- [ ] Select the text column (what they wrote)
- [ ] Choose extraction method:
  - News/general content? → Named Entities
  - Twitter data? → Hashtags or Mentions
  - Want topics? → Keywords
- [ ] Leave other settings as default (unless you know what they do)
- [ ] Click "Start Processing"
- [ ] Wait for it to finish (don't close browser!)
- [ ] Scroll down to see results
- [ ] Explore the visualization (hover, zoom, drag)
- [ ] Download GEXF file to save your work
- [ ] Done!

---

## Remember

**Don't be afraid to experiment!**
- You can't break anything
- Try different extraction methods
- Play with the settings
- Process the same data multiple times

**Start simple:**
- Small file first
- Basic extraction method
- Default settings
- Build up from there

**Have fun exploring your network!**

The tool is designed to help you discover patterns and connections you might not see otherwise. Enjoy the journey of exploring your data!

---

**Questions? Ask the person who set up the app for you.**

**Happy exploring! 🎉**

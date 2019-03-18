# Using code from https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059 by Syed Sadat Nazrul

import requests
import lxml.html as lh
import pandas as pd
import urllib

url='https://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=combined&year_min=1920&year_max=2018&season_start=1&season_end=-1&pos%5B%5D=p&draft_year_min=1936&draft_year_max=2018&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=pick_overall&conference=any&draft_pos%5B%5D=qb&draft_pos%5B%5D=rb&draft_pos%5B%5D=wr&draft_pos%5B%5D=te&draft_pos%5B%5D=e&draft_pos%5B%5D=t&draft_pos%5B%5D=g&draft_pos%5B%5D=c&draft_pos%5B%5D=ol&draft_pos%5B%5D=dt&draft_pos%5B%5D=de&draft_pos%5B%5D=dl&draft_pos%5B%5D=ilb&draft_pos%5B%5D=olb&draft_pos%5B%5D=lb&draft_pos%5B%5D=cb&draft_pos%5B%5D=s&draft_pos%5B%5D=db&draft_pos%5B%5D=k&draft_pos%5B%5D=p&c5val=1.0&order_by=player'

# Create a handle, page, to handle the contents of the website
page = requests.get(url)

# Store the contents of the website under doc
doc = lh.fromstring(page.content)


# Get names of the columns, they're in the second row you take
tr_elements = doc.xpath('//tr')

# Create empty list
col=[]
i=0
# For each row, store each first element (header) and an empty list
for t in tr_elements[1]:
    i+=1
    name=t.text_content()
    name = name + "_" + str(i)
    #print(i,name)
    col.append((name,[]))



# Since out first row is the header, data is stored on the second row onwards
for j in range(2, len(tr_elements)):
    # T is our j'th row
    T = tr_elements[j]

    # If row is not of the same size, the //tr data is not from our table
    if len(T) != len(tr_elements[2]):
        break

    # i is the index of our column
    i = 0

    # Iterate through each element of the row ignoring rows that are titles
    if T[0].text_content() != "Rk":
        for t in T.iterchildren():
            data = t.text_content()
            # Check if row is empty
            if i > 0:
                # Convert any numerical value to integers
                try:
                    data = int(data)
                except:
                    pass
            # Append the data to the empty list of the i'th column
            col[i][1].append(data)
            # Increment i for the next column
            i += 1



##############################

k = 100
while (True):

    temp_url = url + '&offset=' + str(k)

    # Create a handle, page, to handle the contents of the website
    page = requests.get(temp_url)

    # Store the contents of the website under doc
    doc = lh.fromstring(page.content)


    # Get names of the columns, they're in the second row you take
    tr_elements = doc.xpath('//tr')

    if (len(tr_elements) == 0):
        break

    # Since out first row is the header, data is stored on the second row onwards
    for j in range(2, len(tr_elements)):
        # T is our j'th row
        T = tr_elements[j]

        # If row is not of the same size, the //tr data is not from our table
        if len(T) != len(tr_elements[2]):
            break

        # i is the index of our column
        i = 0

        # Iterate through each element of the row ignoring rows that are titles
        if T[0].text_content() != "Rk":
            for t in T.iterchildren():
                data = t.text_content()
                # Check if row is empty
                if i > 0:
                    # Convert any numerical value to integers
                    try:
                        data = int(data)
                    except:
                        pass
                # Append the data to the empty list of the i'th column
                col[i][1].append(data)
                # Increment i for the next column
                i += 1

    k+=100

############################



Dict = {title:column for (title,column) in col}
df = pd.DataFrame(Dict)
df.head()
df.to_csv("P.csv", encoding='utf-8', index=False)



#print(len(tr_elements))

print("code ran")

